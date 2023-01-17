#importing the usual libraries for EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
#loading the data into a DataFrame
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#taking a peek into the DataFrame
df.head()
#Getting some more information about the dataset
df.describe().transpose()
#checking for null/missing values
df.isnull().sum()
df.info()

#seems that there are some categorical columns in this df, let's explore them
#Exploring the target feature
df['Attrition'].unique()
#let's assign 1s and 0s to the Attrition column
df['Attrition'].replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
#Assigning categorical features to 'categorical_cols'
categorical_cols = []
for col, value in df.iteritems():
    if value.dtype == 'object':
        categorical_cols.append(col)
#storing these columns in a new dataframe called df_cat
df_cat = df[categorical_cols]
df_cat.head()
#taking a peek at the unique values in each of the categorical columns
for column in categorical_cols:
    print(f"{column} : {df[column].unique()}")
    print("-"*40)
#assigning numerical variables to our categorical data through sklearn's LabelEncoder
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for column in categorical_cols:
    df[column] = label.fit_transform(df[column])
#checking our new DataFrame with numerical values
df.head()
df.hist(figsize=(20, 20));
#plotting some countplots, splitting on Attrition
plt.figure(figsize=(20,20))

plt.subplot(421)
sns.countplot(x='Age',data=df,hue='Attrition')
plt.subplot(422)
sns.countplot(x='OverTime', data=df, hue='Attrition')
plt.subplot(423)
sns.countplot(x='MaritalStatus', data=df, hue='Attrition')
plt.subplot(424)
sns.countplot(x='JobRole', data=df, hue='Attrition')
plt.subplot(425)
sns.countplot(x='JobLevel', data=df, hue='Attrition')
plt.subplot(426)
sns.countplot(x='JobSatisfaction', data=df, hue='Attrition')
plt.subplot(427)
sns.countplot(x='TotalWorkingYears', data=df, hue='Attrition')
plt.subplot(428)
sns.countplot(x='WorkLifeBalance', data=df, hue='Attrition')

plt.show()
#let's get rid of the StandardHours, EmployeeCount and Over18 column, as all rows have the same value.

df.drop(['StandardHours','Over18','EmployeeCount','EmployeeNumber'],axis=1,inplace=True)
#this will be quite a large heatmap, but will be worth taking a look at to spot correlated features
plt.figure(figsize=(25,25))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
#Splitting the dataset
df_final = df.drop('Attrition',axis=1)
y = df['Attrition']
# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(df_final)
#import the train_test_split model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=71)
#importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
#initializing the RFC object
rfc = RandomForestClassifier(n_estimators=1000)
#fitting the data
rfc.fit(X_train,y_train)
#making the predictions
predictions = rfc.predict(X_test)
#importing some reporting tools
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('classification report: ')
print('='*40)
print(classification_report(y_test,predictions))
print('\n')
print('confusion matrix: ')
print('='*40)
print(confusion_matrix(y_test,predictions))
from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=71)
sm = SMOTE(random_state=71)
X_train, y_train = sm.fit_sample(X_train, y_train)
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print('classification report: ')
print('='*40)
print(classification_report(y_test,predictions))
print('\n')
print('confusion matrix: ')
print('='*40)
print(confusion_matrix(y_test,predictions))