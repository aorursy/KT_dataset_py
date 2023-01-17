#Python Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import pylab as pl

import matplotlib.pyplot as plt

import math

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
#Load Dataset

train1 = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

frames = [train1, test]

train = pd.concat(frames)
#training dataset 

train.head()
#info of training dataset

train.info()
# Survival count comparison

sns.countplot(x ="Survived", data = train)
#Sex comparison in survival count

sns.countplot(x ="Survived",hue ="Sex", data = train)
#Age comparison among passengers

train["Age"].plot.hist()
#Fare Comparison

train["Fare"].plot.hist()
#Dropping columns

train.drop(['Name','Ticket','Cabin'],axis = 1,inplace = True)
# droping all nan values

train.dropna(inplace=True)
#Turning embarked elements into quantitative variable

embarked =pd.get_dummies(train['Embarked'], drop_first = True)

embarked.head()
#Turning embarked elements into quantitative variable

pcl =pd.get_dummies(train['Pclass'], drop_first = True)

pcl.head()
#Turning Sex elements into quantitative variable

sex = pd.get_dummies(train['Sex'], drop_first = True)

sex.head()
#Concatenating quantized dummy variables

train = pd.concat([train,embarked,pcl,sex],axis = 1)
#Modified training dataset

train.head()
train.drop(['Embarked','Pclass','Sex'],axis = 1,inplace = True)
#Filling NaN elements with mean

train.fillna(train.mean(), inplace=True)
#Assigning input and output variables

X = train.drop(['Survived','PassengerId'],axis = 1)

y = train['Survived']
#Splitting training dataset into training and validation dataset

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=123)
#Using Gradient Boost Classifier

clf = GradientBoostingClassifier()

clf.fit(X_train,y_train)
#Predicting output

y_pred = clf.predict(X_valid)
#Classification Report

report = classification_report(y_valid,y_pred)

print(report)
#Accuracy score for Gradient Boost Classifier

accuracy_score(y_valid,y_pred)
#test dataset

test.head()
#test dataset info

test.info()
sex1 = pd.get_dummies(test['Sex'], drop_first = True)
#Modifications in test dataset

embarked1 = pd.get_dummies(test['Embarked'], drop_first = True)

pcl1 = pd.get_dummies(test['Pclass'], drop_first = True)
test = pd.concat([test,embarked1,pcl1,sex1], axis = 1)
test.drop(['Sex', 'Embarked', 'Name', 'Ticket','Pclass','Cabin'], axis=1,inplace=True)
#Filling test variables

test.fillna(test.mean(), inplace=True)
# Modified test dataset

test.head()
test.info()
# submit your predictions in csv format

ids = test['PassengerId']

predictions = clf.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)