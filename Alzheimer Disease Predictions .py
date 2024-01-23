#!/usr/bin/env python
# coding: utf-8

# # Alzheimers Disease Prediction
# 
# ## Comparitative Analysis of Various Machine Learning Algorithms 

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chisquare


# In[2]:


## Read the dataset
df = pd.read_csv('oasis_longitudinal.csv')

## Display the summary of the dataset
df.info()


# In[3]:


# COL	Description
# EDUC	Years of Education
# SES	Socioeconomic Status
# MMSE	Mini Mental State Examination
# CDR	Clinical Dementia Rating
# eTIV	Estimated Total Intracranial Volume
# nWBV	Normalize Whole Brain Volume
# ASF	Atlas Scaling Factor

print("Total No of Rows and Columns (Rows,Columns) : ",df.shape)
df.head(10)


# In[4]:


# Statisitical Analysis of the Dataset
df.describe()


# In[5]:


#No of rows and columns containing null values
df.isna().sum()


# In[6]:


# No of duplicate entries
sum(df.duplicated())


# In[7]:


sns.set_style("whitegrid")
ex_df = df.loc[df['Visit'] == 1]
sns.countplot(x='Group', data=ex_df)
ex_df['Group'] = ex_df['Group'].replace(['Converted'], ['Demented'])
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
sns.countplot(x='Group', data=ex_df)


# In[8]:


# Socio Economic Status (SES) and Mini Mental State Examination (MMSE) contains null vallues
# Fill these null values with mean and median values 
columns=['Visit','MR Delay','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF']
df["SES"].fillna(df["SES"].median(), inplace=True)
df["MMSE"].fillna(df["MMSE"].mean(), inplace=True)


# In[9]:


# Mean Values of the variables
print("Mean:")
df[columns].mean()


# In[10]:


# Median Values of the variables
print("Median: ")
df[columns].median()


# In[11]:


# Mode Values of the variables
print("Mode: ")
df[columns].mode()


# In[12]:


# F Test (ANOVA Analysis)

# 'Group' is the categorical column (Demented, Non-demented) 
# and 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF' are your numerical columns

# Perform one-way ANOVA for each numerical column
for column in ['EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']:
    fvalue, pvalue = stats.f_oneway(df[column][df['Group'] == 'Demented'],
                                    df[column][df['Group'] == 'Nondemented'])
    print(f'\n{column}:')
    print('F-value:', fvalue)
    print('P-value:', pvalue)


# In[13]:


# Perform t-test for each numerical column
for column in ['EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']:
    group1 = df[column][df['Group'] == 'Demented']
    group2 = df[column][df['Group'] == 'Nondemented']
    
    t_statistic, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
    
    print(f'\n{column}:')
    print('t-statistic:', t_statistic)
    print('P-value:', p_value)


# In[14]:


# Chi-Square Test for Indepependence 

# Create a contingency table
contingency_table = pd.crosstab(df['Group'], df['M/F'])

# Perform Chi-Square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(contingency_table)


# In[15]:


print('Chi-square statistic:', chi2)
print('P-value:', p_value)


# In[16]:


# Assuming df is your DataFrame and 'Group' is your categorical column

# Get observed frequencies
observed = df['Group'].value_counts()

# Define expected frequencies (assuming equal distribution)
expected = [len(df)/len(observed) for _ in range(len(observed))]

# Perform Chi-square test
chi2, p = chisquare(observed, f_exp=expected)

# Create a DataFrame for observed and expected frequencies
freq_df = pd.DataFrame({
    'Observed': observed,
    'Expected': expected
})

print("\nObserved and Expected Frequencies:")
print(freq_df)
print("\n\n")
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")


# In[17]:


#Importing the cross-sectional CSV data from the OASIS (Alzheimers) Study: 

crossSectionalMRI = pd.read_csv("oasis_cross-sectional.csv")
crossSectionalMRI.head()



# In[18]:


#Supervised Learning Algorithms 

# 1. Linear Regression 

#Importing the Linear Regression Package from Scikit Learn.  
from sklearn.linear_model import LinearRegression 

AgeAndCDRLinearReg = LinearRegression()

#Because there are a number of NaN values in the CDR column, all of the subjects with NaN CDR values will be dropped from the following correlation. 
crossSectionalMRI.dropna(subset = ["CDR","SES","Educ","MMSE"], axis = 0, inplace = True)
crossSectionalMRI.head()

Age = crossSectionalMRI[['Age']]
CDRScores = crossSectionalMRI[['CDR']]

AgeAndCDRLinearRegModel = AgeAndCDRLinearReg.fit(Age,CDRScores)

AgeAndCDRLinearRegModel.score(Age,CDRScores)


# In[19]:


#The correlation between Age and CDR scores is actually extremely low within this sample of Alzheimer's patients. Pearson's Coefficient in this case is only ~0.09.


# In[20]:


import seaborn as sns 
from matplotlib import pyplot as plt
sns.regplot(x='Age',y='CDR',data=crossSectionalMRI)
plt.ylim(0)


# In[23]:


#2 logistic Regression
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Let's check the datatype of the target variable column (CDR): 
before = crossSectionalMRI.dtypes
before
#This column contains data of the type: float64. Let's change that for scikit learn compatability. 

#Let's convert the target data type to integer (as required by scikit learn): 
crossSectionalMRI['CDR'] = crossSectionalMRI['CDR'].astype(int)

#Let's check to ensure the datatype was correctly changed: 
after = crossSectionalMRI.dtypes
after

#Let's convert the Pandas dataframe above into two numpy arrays for more ease of use with scikit learn functions (train/test splitting, etc.): 
crossSectionalMRIFeatures = np.asarray(crossSectionalMRI[['Age','SES','Educ']])
crossSectionalMRITarget = np.asarray(crossSectionalMRI['CDR'])

#Next, let's split the whole dataset into training and testing sets for higher validity: 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(crossSectionalMRIFeatures, crossSectionalMRITarget, test_size=0.2, random_state=4)

#Let's build the Multiple Logistic Regression Model using the training sets 
#and compute some relevant metrics and, perhaps, make a few predictive statements: 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
crossSectionalMRILogistic = LogisticRegression(C=0.01, solver='newton-cg', multi_class='multinomial').fit(X_train,y_train)
crossSectionalMRILogistic

#Finally, let's make a few predictions using this model and the test set, as well as the probability of each of the class targets (0, 0.5, and 1 CDR Scores): 
LogisticAlzhemiersCDRScorePreds = crossSectionalMRILogistic.predict(X_test) 
LogisticAlzhemiersCDRScorePreds


# In[24]:


LogisticAlzheimersCDRScoreProbas = crossSectionalMRILogistic.predict_proba(X_test)
LogisticAlzheimersCDRScoreProbas


# In[ ]:





# In[25]:


#Issues with logisitc model through confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

#The following chunk of code (the rest of this code cell)
#was borrowed from the Coursera IBM Professional Certificate Course on Machine Learning (Course 8):

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, LogisticAlzhemiersCDRScorePreds, labels=[1,0]))

AlzheimersLogisticConfusionMatrix = confusion_matrix(y_test, LogisticAlzhemiersCDRScorePreds, labels=[1,0.5,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(AlzheimersLogisticConfusionMatrix, classes=['CDR=1','CDR=0.5','CDR=0'], normalize= False,  title='Alzheimers Logistic CDR Score Confusion Matrix')


# In[26]:


# Encode columns into numeric
from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


# In[27]:


from sklearn.model_selection import train_test_split

feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
predicted_class_names = ['Group']

X = df[feature_col_names].values
y = df[predicted_class_names].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[28]:


from sklearn import metrics
def plot_confusion_metrix(y_test,model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Nondemented','Demented']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[29]:


from sklearn.metrics import roc_curve, auc
def report_performance(model):

    model_test = model.predict(X_test)

    print("\n\nConfusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    #cm = metrics.confusion_matrix(y_test, model_test)
    plot_confusion_metrix(y_test, model_test)

def roc_curves(model):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test,y_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[30]:


total_models = ['SVC','DecisionTreeClassifier','KNeighborsClassifier','LogisticRegression']
total_accuracy = {}
for i in total_models:
    total_accuracy[str(i)] = 0
def accuracy(model):
    pred = model.predict(X_test)
    accu = metrics.accuracy_score(y_test,pred)
    print("\nAccuracy Of the Model: ",accu,"\n\n")
    total_accuracy[str((str(model).split('(')[0]))] = accu


# In[31]:


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# In[32]:


clf_dtc = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
clf_dtc.fit(X_train, y_train.ravel())
report_performance(clf_dtc) 
roc_curves(clf_dtc)
accuracy(clf_dtc)


# In[33]:


clfs =[LogisticRegression()]
for model in clfs:
    print(str(model).split('(')[0],": ")
    model.fit(X_train,y_train.ravel())
    X = pd.DataFrame(X_train)
    report_performance(model)
    roc_curves(model)
    accuracy(model)


# In[ ]:





# In[34]:


svm = SVC(kernel="linear", C=0.1,random_state=0)
svm.fit(X_train, y_train.ravel())
report_performance(svm) 
roc_curves(svm)
accuracy(svm)


# In[35]:


rfc=RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [200],
    'max_features': ['auto'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5,scoring = 'roc_auc')
CV_rfc.fit(X_train, y_train.ravel())
print("Best parameters set found on development set:")
print(CV_rfc.best_params_)
report_performance(CV_rfc) 
roc_curves(CV_rfc)
accuracy(CV_rfc)


# In[36]:


clfs =[KNeighborsClassifier()]
for model in clfs:
    print(str(model).split('(')[0],": ")
    model.fit(X_train,y_train.ravel())
    X = pd.DataFrame(X_train)
    report_performance(model)
    roc_curves(model)
    accuracy(model)


# In[37]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load your dataset
data = pd.read_csv("oasis_longitudinal.csv")  # Replace with the actual dataset file

# Data preprocessing
# Drop any rows with missing values or perform data imputation as needed
data = data.dropna()

# Select the features and target variable
X = data[['Visit', 'MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
y = data['CDR']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = keras.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Make predictions
predictions = model.predict(X_test)

# You can use the predictions to make Alzheimer's disease predictions.


# 

# In[38]:


# # Assuming your original y_test contains continuous values
# threshold = 0.5  # Define a threshold for binary classification
# binary_y_test = (y_test >= threshold).astype(int)

# # Compute the confusion matrix
# confusion = confusion_matrix(binary_y_test, predicted_labels)

# # Print the confusion matrix
# print("Confusion Matrix:")
# print(confusion)


# In[39]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Alzheimer's dataset
data = pd.read_csv("oasis_longitudinal.csv")

# Select relevant columns for clustering
selected_columns = ["Age", "EDUC", "SES", "MMSE", "nWBV", "ASF"]

# Remove rows with missing values
data = data.dropna(subset=selected_columns)

# Standardize the selected features
scaler = StandardScaler()
data[selected_columns] = scaler.fit_transform(data[selected_columns])

# Determine the number of clusters (k) using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[selected_columns])
    inertia.append(kmeans.inertia_)

# Plot the Elbow method to select the optimal k
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Based on the Elbow method, select an appropriate k (number of clusters)

# Perform k-means clustering with the chosen k
k = 3  # You can change this based on the Elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[selected_columns])

# Explore the resulting clusters
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_columns)
cluster_sizes = data['Cluster'].value_counts().sort_index()

# Display cluster centers and sizes
print("Cluster Centers:")
print(cluster_centers)
print("\nCluster Sizes:")
print(cluster_sizes)



# In[40]:


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the Alzheimer's dataset
data = pd.read_csv("oasis_longitudinal.csv")

# Exclude non-numeric columns like 'M/F' and 'Group'
numeric_features = data.select_dtypes(include=['int64', 'float64'])

# Drop rows with missing values (NaN)
numeric_features = numeric_features.dropna()

# Standardize the feature matrix (mean=0, variance=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

# Perform PCA with 2 components
n_components = 2
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame to store the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Alzheimer\'s Dataset with 2 Components')

plt.show()


# In[ ]:




