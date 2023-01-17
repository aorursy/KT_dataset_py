# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load data 
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/gender_submission.csv')

train_data.head()
# data synopsis
train_data.info()
train_data.describe()
### imputing missing value ###
print(train_data.isnull().sum())
test_data.isnull().sum()
sns.distplot(train_data.loc[train_data.Age.notnull(), 'Age'])

# fill nan with  the age value corresponding to the highest, the most frequently
train_data.Age.fillna(value=train_data.Age.mode()[0],inplace=True)
test_data.Age.fillna(value=test_data.Age.mode()[0], inplace=True)
# imputing Embarked column
train_data.Embarked.fillna(value=train_data.Embarked.mode()[0], inplace=True)
test_data.Embarked.fillna(value=test_data.Embarked.mode()[0], inplace=True)
# imputing Cabin
# since too many missing value, we just replace with a notation
train_data.Cabin.fillna(value='Unknown', inplace=True)
test_data.Cabin.fillna(value='Unkown', inplace=True)

# imputing Fare
test_data.Fare.fillna(value=test_data.Fare.mean(), inplace=True)

# see if all imputed
print(train_data.isnull().sum())
test_data.isnull().sum()
### feature engineering: dummpy encode ###

data_set = [train_data, test_data]
sex_mapping = {'female': 0, 'male': 1}
train_data.Embarked.unique()
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

for data in data_set:
    data.Sex = data.Sex.map(sex_mapping)
    data.Embarked = data.Embarked.map(embarked_mapping)

# Cabin has too much missing information, I dare to drop it
for data in data_set:
    data.drop(columns='Cabin', inplace=True)

# What about Name?
# We create a Title column generated from Name, then drop it
for data in data_set:
    data['Title'] = data.Name.str.extract(r'([a-zA-Z]+)\.', expand=False)
    data.drop('Name', axis=1, inplace=True)
## aggregate Title, 'Other' represents the minority 
for data in data_set:
     data.Title = data.Title.map(lambda x : 'Other' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)

## mapping Title
print(train_data.Title.unique())
print(test_data.Title.unique())
title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other':4}
for data in data_set:
    data.Title = data.Title.map(title_mapping)

## Analysis SibSp, Parch
for data in data_set:
    data['Families'] = data.SibSp + data.Parch + 1
    data['Alone'] = data.Families.map(lambda x: 0 if x > 1 else 1)
    data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test_data.head()
train_data.head()
## is Ticket any valuable hint?
# maybe, but it's too noisy
for data in  data_set:
    data.drop('Ticket', axis=1, inplace=True)
    data.drop('PassengerId', axis=1, inplace=True)
## Now it looks cleaner
train_data.head()
## Visual Statistic
sns.countplot(x='Title', hue='Survived', data=train_data)
sns.countplot(x='Sex', hue='Survived', data=train_data)
## Let's modeling
# import model lib
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import  RandomForestClassifier as RFC
from xgboost import XGBClassifier as XGBC

# import mopdel selection
from sklearn.model_selection import train_test_split
# quality validation
from sklearn.metrics import  accuracy_score

target = 'Survived'
y = train_data[target]
X = train_data.drop(target,axis=1)
train_X, test_X, train_y, test_y = train_test_split(X, y)
model_score = {}
# 1. decision tree
dtc = DTC(random_state=1)
dtc.fit(train_X, train_y)
prediction_dtc = dtc.predict(test_X)
model_score[dtc] = accuracy_score(test_y, prediction_dtc)
# 2. RandomForest Classifier
rfc = RFC(random_state=1)
rfc.fit(train_X, train_y)
prediction_rfc = rfc.predict(test_X)
model_score[rfc] = accuracy_score(test_y, prediction_rfc)
# 3. XGBC
params = {'eta': 0.5, 'max_depth':6, 'gamma': 1, 'subsample': 1, 'reg_alpha': 1, 
          'n_jobs': -1, 'random_state': 1, 'n_estimators': 100}
xgbc = XGBC(**params)
xgbc.fit(train_X, train_y, early_stopping_rounds=3, eval_set=[[test_X, test_y]])
prediction_xgbc = xgbc.predict(test_X)
score = accuracy_score(test_y, prediction_xgbc)
# print('*'*30)
# print(score)

real_prediction = xgbc.predict(test_data)
submission = pd.DataFrame({'PassengerId': sample.PassengerId, 'Survived': real_prediction})
submission.to_csv('submission.csv', index=False)