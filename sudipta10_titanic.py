#Import Libraries



import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
#Import Train and Test Data



df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')

y_test=pd.read_csv('../input/titanic/gender_submission.csv')

pandas_profiling.ProfileReport(df_train)
df_train.isnull().sum()
df_test.isnull().sum()
#Impute Missing value--Train Data

df_train['Age'].fillna(df_train.Age.mean(), inplace=True)

df_train['Embarked'].fillna(df_train.Embarked.mode()[0], inplace=True)

df_train.isnull().sum()
#Impute Missing value--Test Data 

df_test['Age'].fillna(df_test.Age.mean(), inplace=True)

df_test['Fare'].fillna(df_test.Fare.mean(), inplace=True)

df_test['Embarked'].fillna(df_test.Embarked.mode()[0], inplace=True)

df_test.isnull().sum()
#Removing Coloumn Cabin from Train Data & Test Data which contains much null value and this coloumn is not so useful

df_train = df_train.drop('Cabin', axis=1)

df_test = df_test.drop('Cabin', axis=1)
y_train = df_train['Survived']

y_train.head()
x_train = df_train.iloc[:,2:]

x_train = x_train.drop(['Name','Ticket'], axis=1)

x_train.head()
x_train.dtypes
#Feature Engineering

x_train['Embarked'] = x_train['Embarked'].replace('S',1)

x_train['Embarked'] = x_train['Embarked'].replace('C',2)

x_train['Embarked'] = x_train['Embarked'].replace('Q',3)

x_train['Sex'] = x_train['Sex'].replace('male',1)

x_train['Sex'] = x_train['Sex'].replace('female',2)
x_train.head()
lr = LogisticRegression()
lr.fit(x_train, y_train)
df_test['Embarked'] = df_test['Embarked'].replace('S',1)

df_test['Embarked'] = df_test['Embarked'].replace('C',2)

df_test['Embarked'] = df_test['Embarked'].replace('Q',3)

df_test['Sex'] = df_test['Sex'].replace('male',1)

df_test['Sex'] = df_test['Sex'].replace('female',2)
x_test=df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

x_test.head()
y_pred = lr.predict(x_test)
y_test.head()
accuracy = accuracy_score(y_test['Survived'], y_pred)

f1 = f1_score(y_test['Survived'], y_pred)

precision = precision_score(y_test['Survived'], y_pred)

recall = recall_score(y_test['Survived'], y_pred)

roc_auc = roc_auc_score(y_test['Survived'], y_pred)
print('Accuracy is  :' ,accuracy)

print('F1 score is :' ,f1)

print('Precision is  :',precision)

print('Recall is  :',recall)

print('Roc Auc is  :',roc_auc)
PassengerId=y_test['PassengerId']

Survived  = pd.Series(y_pred)

submission = pd.concat([PassengerId,Survived], axis=1)

submission.columns = ["PassengerId", "Survived"]

submission.head()
submission.to_csv('submission.csv',index=False)