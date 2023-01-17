# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
%matplotlib inline 

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import seaborn as sns


#Importing models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing


#Metrices
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

original_df=pd.read_csv("../input/bank.csv")
#Copy the original dataset into df_bank.It will be good if we need to clean the data in later stages.
df_bank=original_df.copy()
df_bank.head()

#What are the total number of records available ?
df_bank.shape
#We can see that total number of records are 11,162 and total number of attributes per record is 17
#Now we wil see the short desriptive summary of all the bank customer data
df_bank.describe()
# Pls note that we have only 7 numerical attributes.So only these will be considered here.
# Most of our data is in Categorical Column


#df_bank.columns
#First use the pairplot to understand the relationship
plt.figure(figsize=(17,10))
sns.pairplot(df_bank,hue='deposit')
#Plot the Correlation in a Heatmap
corr_matrix=df_bank.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix,annot=True)
#Lets plot the Age Distribution
df_training.age.hist(bins=20, figsize=(14,10), color='#E14906')
plt.show()
# What is the percentage of Customer that have subscribed to term deposit vs not deposit
# What is the Percentage of Cutomer Marital Status in our Data that has subscribed to Term Deposit ?

f, ax = plt.subplots(1,2, figsize=(10,6))
plt.suptitle('Term Deposit Status')

df_deposit=df_bank.deposit.value_counts()
#print(df_deposit)
ax[0].pie(df_deposit, labels=['Opened','Not Opened'], autopct='%1.1f%%',
         explode=(0.10,0),shadow=True, startangle=60)
ax[0].set_xlabel('Opened vs Not Opened', fontsize=14)



#Part 2
df_marital=df_bank[df_bank['deposit']=='yes'].marital.value_counts()
#print(df_marital)
#Labels To Be used
ax[1].pie(df_marital, labels=df_marital.index, autopct='%1.1f%%',
         explode=(0.10,0,0),shadow=True, startangle=60)
ax[1].set_xlabel('Opened Status based on Martial Status', fontsize=14)


plt.axis('equal')
plt.tight_layout()
plt.show()
f, ax = plt.subplots(1,3, figsize=(15,8))
plt.suptitle('Term Deposit Status Based On Martial Status')

#Part 1
df_martial_status=df_bank.marital.value_counts()
#print(df_martial_status)
ax[0].pie(df_martial_status, labels=df_martial_status.index, autopct='%1.1f%%',
         explode=(0.05,0,0),shadow=True, startangle=60)
ax[0].set_xlabel('All Term Deposit', fontsize=14)

#Part 2
df_marital_yes=df_bank[df_bank['deposit']=='yes'].marital.value_counts()
#print(df_marital_yes)
#Labels To Be used
ax[1].pie(df_marital_yes, labels=df_marital_yes.index, autopct='%1.1f%%',
         explode=(0.05,0,0),shadow=True, startangle=60)
ax[1].set_xlabel('Opened Term Deposit Martial Status', fontsize=14)


#Part 3
df_marital_no=df_bank[df_bank['deposit']=='no'].marital.value_counts()
#print(df_marital_no)
#Labels To Be used
ax[2].pie(df_marital_no, labels=df_marital_no.index, autopct='%1.1f%%',
         explode=(0.05,0,0),shadow=True, startangle=60)
ax[2].set_xlabel('Closed Term Deposit Martial Status', fontsize=14)

plt.axis('equal')
plt.tight_layout()
plt.show()
plt.figure(figsize=(18,4))
#print(df_bank.columns)
print(df_bank.job.value_counts())
#Count Plot (a.k.a. Bar Plot)
ax=sns.countplot(x='job', hue='deposit',data=df_bank).set_title('Customer Job Distribution')
plt.xlabel('Job Category')
plt.ylabel('Count of Jobs')
plt.xticks(rotation=-30)






df_training=original_df.copy()
df_training.head()
#First check what all are the columns present
df_training.columns
#Let us examine the dataset for null values
df_training.isnull().sum()
#Create a list of all the features needed in our model
used_features   = ['job', 'marital', 'default', 'housing', 'loan', 'poutcome']
#Prepare our Dataset for features and target variable
df_X = df_training[used_features]
df_y = df_training['deposit']
#Print the Shape of both dataset
print(df_X.shape)
print(df_y.shape)
#Convert our categorical features into DUmmy variables for One Hot Encoding
df_training_dummied = pd.get_dummies(df_X, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
df_training_dummied.columns
# Check the independence between the independent variables
df_training_dummied.corr()
#plt.figure(figsize=(15,10))
#sns.heatmap(df_training_dummied.corr(),annot=True)


#print(df_y)
#Convert the Target variable to 1=Yes and 0 = No
le = preprocessing.LabelEncoder()
# train the encoder with the label data
#Transform
df_ytransformed=le.fit_transform(df_y)
df_ytransformed

# Split the data into training and test sets
X = df_training_dummied.values
y = df_ytransformed
print(X.shape)
print(y)
#Divide the Dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#Instatiate the classifier and fit the model
simple_classifier = LogisticRegression(random_state=42,solver='liblinear')
simple_classifier.fit(X_train, y_train)
#simple_classifier.classes_
y_pred = simple_classifier.predict(X_test)
#Confusion Matrix
cnf_matrix=metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
#Visualize confusion matrix using heat map

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#ROC Curve is a plot of the true positive rate against the false positive rate.
#It shows the tradeoff between sensitivity and specificity.

y_pred_proba = simple_classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#ROC Curve is a plot of the true positive rate against the false positive rate.
#It shows the tradeoff between sensitivity and specificity.

y_pred_proba = simple_classifier.predict_proba(X_test)[::,0]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 0, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#Check the coefficients for multiple features
print(simple_classifier.coef_)
#Check the intercept
print(simple_classifier.intercept_)


#Using the Cross Validation Method
cv_classifier=LogisticRegression(random_state=42,solver='liblinear')
scoring = ['precision_macro', 'recall_macro', 'f1','accuracy','balanced_accuracy']
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
scores=cross_validate(cv_classifier, X, y, cv=cv,scoring=scoring,return_train_score=False) 
scores

