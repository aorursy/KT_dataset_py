#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
# Print the first seven rows of data
df.head(7)
#Get the count of rows and columns
df.shape
df.head()
# To know the data types 
df.dtypes
# To check for the null values
df.isna().sum()
# To check missing values 
df.isnull().any()
# To view Statistics
df.describe()
# Count of no. of employees stayed and left the company
df['Attrition'].value_counts()
# Visualize attrition
sns.countplot(df['Attrition'])
# Just by guessing "No" everytime the model accuracy will be
(1233-237)/1233
# Attrition analysis based on age
plt.subplots(figsize=(16,6))
sns.countplot(x='Age',hue='Attrition',data=df)
for column in df.columns:
    if df[column].dtype == object:
     print(str(column)+' : '+str(df[column].unique()))
     print(df[column].value_counts())
     print('                   ')
# Remove unwanted columns
df=df.drop('Over18',axis = 1)
df=df.drop('EmployeeNumber',axis = 1)
df=df.drop('StandardHours',axis = 1)
df=df.drop('EmployeeCount',axis = 1)
# To check the correlation
df.corr()
plt.figure(figsize=(16,14))
sns.heatmap(df.corr(), annot = True, fmt = '.0%', cmap='coolwarm')
df.describe(include=['object'])
# Making a frame of all the categorical variables except for Attrition, we will treat attrition column using label encoder

df1=pd.DataFrame(df,columns = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'])

# Drop the particulars columns from main data
df=df.drop('BusinessTravel',axis=1)
df=df.drop('Department',axis=1)
df=df.drop('EducationField',axis=1)
df=df.drop('Gender',axis=1)
df=df.drop('JobRole',axis=1)
df=df.drop('MaritalStatus',axis=1)
df=df.drop('OverTime',axis=1)
df.info()
# Data Transformation using get_dummies and label encoder

from sklearn.preprocessing import LabelEncoder
df['Attrition']= LabelEncoder().fit_transform(df['Attrition'])

dummy=pd.get_dummies(df1)
frame=[df,dummy]

df=pd.concat(frame,axis=1)
df.info()
df.shape
# Seperating the feature set and label set
X=df[df.columns.difference(['Attrition'])]
Y=df['Attrition']
# Split the data into training set (70%) and test set (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=0)
# Using Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 30, criterion = 'gini', random_state =1, max_depth = 10)
forest.fit(X_train, Y_train)
# Apply the model on X_test
pred_forest = forest.predict(X_test)
# Finding Accuracy
from sklearn.metrics import accuracy_score, f1_score
forest = accuracy_score(Y_test,pred_forest)*100
# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(pred_forest, Y_test)
print (cm)
print(classification_report(pred_forest, Y_test))
