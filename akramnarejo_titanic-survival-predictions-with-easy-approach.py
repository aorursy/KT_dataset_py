# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



# warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# some information about the dataframe

train.info()
# Data types of columns

train.dtypes
# Summary

train.describe() # by default we get the summary of numerical datatype columns
# Sex feature

sns.barplot(x=train.Sex, y=train.Survived)

print(f'females survival percent: ',train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print(f'males survival percent: ',train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
# Pclass feature

sns.barplot(x=train.Pclass, y=train.Survived)

print(f'Pclass 1 survival percent: ',train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print(f'Pclass 2 survival percent: ',train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print(f'Pclass 3 survival percent: ',train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
# checking missing values in train dataset

train.isnull().sum()
# checking missing values in test dataset

test.isnull().sum()
# Along with cabin columns we also need to remove a few more columns which will be meaningless in future like PassengerId, Name, Ticket, Cabin

train.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

test.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
# Replace the Sex column male with 1 and female with 0 in train and test datasets

train['Sex'] = [1 if gender=='male' else 0 for gender in train['Sex']]

test['Sex'] = [1 if gender=='male' else 0 for gender in test['Sex']]
train.head()
train.info()
# filling the missing age values in train and test

train.Age.fillna(train.Age.median(), inplace=True)

test.Age.fillna(test.Age.median(), inplace=True)
# grouping the age in train and test

bins = [0,10,20,40,60,80,100]

groups = [0,2,3,4,5,6]

train['AgeGroup'] = pd.cut(train['Age'], bins, labels=groups, include_lowest=True)

test['AgeGroup'] = pd.cut(test['Age'], bins, labels=groups, include_lowest=True)
# so remove the age column from both train and test

train.drop(columns=['Age'], inplace=True)

test.drop(columns=['Age'], inplace=True)
# there is a missing value in the Fare column of test dataset so let's fill that first

test.Fare.fillna(test.Fare.median(), inplace=True)
# Now lets group the fare column for both train and test dataset

bins = [0,100,200,300,400,500,600]

groups = [0,1,2,3,4,5]

train['FareGroup'] = pd.cut(train['Fare'], bins, labels=groups, include_lowest=True)

test['FareGroup'] = pd.cut(test['Fare'], bins, labels=groups, include_lowest=True)
# Now remove the Fare column from both datasets

train.drop(columns=['Fare'], axis=1, inplace=True)

test.drop(columns=['Fare'], axis=1, inplace=True)
# we have missing values in Embarked column in train dataset so lets fill that first

train.Embarked.fillna(train.Embarked.mode(), inplace=True)
train = pd.get_dummies(train, columns=['Pclass','AgeGroup','Embarked'])

test = pd.get_dummies(test, columns=['Pclass','AgeGroup','Embarked'])
train.head()
predictors = train.drop(columns=['Survived'], axis=1)
target = train[['Survived']]
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(predictors,target,test_size=0.2,random_state=123)
# lets see the shapes

print(x_train.shape)

print(x_val.shape)

print(y_train.shape)

print(y_val.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC
# importing accuracy 

from sklearn.metrics import accuracy_score
lr = LogisticRegression()

lr.fit(x_train,y_train)

preds = lr.predict(x_val)

lr_accuracy = accuracy_score(y_val,preds)

print(f'Logistic Regression accuracy: {lr_accuracy*100}')
knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

preds = knn.predict(x_val)

knn_accuracy = accuracy_score(y_val, preds)

print(f'KNN accuracy: {knn_accuracy*100}')
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

preds = dt.predict(x_val)

dt_accuracy = accuracy_score(y_val, preds)

print(f'Decission Tree accuracy: {dt_accuracy*100}')
rf = RandomForestClassifier()

rf.fit(x_train,y_train)

preds = rf.predict(x_val)

rf_accuracy = accuracy_score(y_val, preds)

print(f'RandomForest accuracy: {rf_accuracy*100}')
gbc = GradientBoostingClassifier()

gbc.fit(x_train,y_train)

preds = gbc.predict(x_val)

gbc_accuracy = accuracy_score(y_val, preds)

print(f'GradientBoostClassifier accuracy: {gbc_accuracy*100}')
svc = SVC()

svc.fit(x_train,y_train)

preds = svc.predict(x_val)

svc_accuracy = accuracy_score(y_val, preds)

print(f'SVC accuracy: {svc_accuracy*100}')
models = pd.DataFrame({'Model':['LogisticRegression','KNN','DecissionTree','RandomForest','GradientBoostClassifier','SVM'],

         'Accuracy':[lr_accuracy*100,knn_accuracy*100,dt_accuracy*100,rf_accuracy*100,gbc_accuracy*100,svc_accuracy*100]})

models
data = pd.read_csv('/kaggle/input/titanic/test.csv')

ids = data['PassengerId']
preds = dt.predict(test)
output = pd.DataFrame({'PassengerId':ids, 'Survived':preds})
output.to_csv('submission.csv', index=False)