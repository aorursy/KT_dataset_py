import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

import math

%matplotlib inline

import os



data= pd.read_csv('../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv')

data
sns.countplot(x='Survived' , data=data)
sns.countplot(x='Survived' , hue='Sex' , data=data)
sns.countplot(x='Survived' , hue='Pclass' , data=data)
data['Age'].plot.hist()
data['Fare'].plot.hist()
data['Fare'].plot.hist(bins= 10 , figsize=(10,5))
data.info()
sns.countplot(x='SibSp',data=data)
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False)
sns.heatmap(data.isnull(),yticklabels=False, cmap='viridis')
sns.boxplot(x='Pclass',y='Age',data=data)
data.drop('Cabin',axis=1,inplace=True)
data.head(5)
data.dropna(inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False)
data.isnull().sum()
data.head(2)
sex=pd.get_dummies(data['Sex'], drop_first=True)

sex.head(2)
embark=pd.get_dummies(data['Embarked'],drop_first=True)

embark.head(5)
pcl=pd.get_dummies(data['Pclass'],drop_first=True)

pcl.head(5)
data=pd.concat([data,sex,embark,pcl],axis=1)

data.head(5)
data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
data.head(5)
x=data.drop('Survived',axis=1)

y=data['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
Predictions = logmodel.predict(X_test)

Predictions
from sklearn.metrics import classification_report
classification_report(y_test,Predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test , Predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test , Predictions)