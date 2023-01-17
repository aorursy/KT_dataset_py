# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from random import *

from collections import *

from statistics import *

from pprint import pprint



from collections import defaultdict



import logging

import datetime



import glob

import csv



import matplotlib as plt

import seaborn as sns

import plotly



import calendar

#calendar.month(2019,2)

#print(calendar.month(2019,2))



pd.set_option('display.max_columns', None)



import tensorflow as tf

from tensorflow import keras



print(f'The current time is',datetime.datetime.now())
myPath = '/kaggle/input/titanic/'



test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

submission0 = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#Transform ln

train_data['lnAge'] = np.log(train_data['Age']+1)

train_data['lnSibSp'] = np.log(train_data['SibSp']+1)

train_data['lnParch'] = np.log(train_data['Parch']+1)

train_data['lnFare'] = np.log(train_data['Fare']+1)



train_data.head()

#Target = Survived

#Pclass

#Name - Miss vs Mrs.

#Sex - 1-hot-encode

#Age - lnAge

#SibSp - lnSibSp

#Parch - lnParch

#Fare - lnFare

#Embarked  - 1-hot-encode
train_data.info()
sns.countplot(train_data['Embarked'])
train_data['Embarked'].value_counts()
g = sns.FacetGrid(train_data, col='Embarked')

g.map(sns.countplot,"Survived")
g = sns.FacetGrid(train_data, col='Sex')

g.map(sns.countplot,"Survived")
g = sns.FacetGrid(train_data, col='Survived')

g.map(sns.kdeplot,"lnFare")
sns.countplot(train_data['Parch'])
sns.countplot(train_data['Pclass'])
sns.countplot(train_data['SibSp'])
sns.lmplot(x='Survived',y='Age',hue='Embarked',data=train_data[train_data['Sex']=='female'])
#Scatter

train_data.sample(100).plot.scatter(x='Survived', y='Age')

sns.pairplot(train_data[['Survived','lnAge','lnFare','Parch','SibSp']])
#Transform ln

test_data['lnAge'] = np.log(test_data['Age']+1)

test_data['lnSibSp'] = np.log(test_data['SibSp']+1)

test_data['lnParch'] = np.log(test_data['Parch']+1)

test_data['lnFare'] = np.log(test_data['Fare']+1)



test_data.head()
#Fill in embarked

train_data['Embarked']=train_data['Embarked'].fillna('S')



train_data['Embarked'].value_counts()
#Fill in embarked

test_data['Embarked']=test_data['Embarked'].fillna('S')



test_data['Embarked'].value_counts()
#One hot encode

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



train_data['Sex'] = le.fit_transform(train_data['Sex'])

train_data['Embarked'] = le.fit_transform(train_data['Embarked'])





test_data['Sex'] = le.fit_transform(test_data['Sex'])

test_data['Embarked'] = le.fit_transform(test_data['Embarked'])

# Y = 'Survived'

# x = 'lnAge','lnFare','Parch','SibSp', 'Sex', 'Embarked' by 'PassengerId'



Y_train = train_data['Survived']

x_train = train_data[['PassengerId','Age','lnFare','Parch','SibSp', 'Sex', 'Embarked']]





x_test = test_data[['PassengerId','Age','lnFare','Parch','SibSp', 'Sex', 'Embarked']]



import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_train, Y_train)

predictions = gbm.predict(x_test)



submission2 = pd.DataFrame({ 'PassengerId': x_test['PassengerId'],

                            'Survived': predictions })

submission2.to_csv("submission2.csv", index=False)



print(submission2)
#Use h2o

import h2o

h2o.init()
#Change to h2o datasets





x_train_h2o = h2o.H2OFrame(train_data[['Survived','PassengerId','Age','lnFare','Parch','SibSp', 'Sex', 'Embarked']])







x_test_h2o = h2o.H2OFrame(x_test)

print(x_train_h2o)
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "Titanic_H2O")

aml.train(y = 'Survived', training_frame = x_train_h2o, leaderboard_frame = x_train_h2o)

aml.leaderboard.head()

pred = aml.predict(x_test_h2o)

pred.head()

perf = aml.leader.model_performance(x_test_h2o)

perf
perf = aml.leader.model_performance(x_train_h2o)

perf
#Submission File

## Prediction of Model 892 to 1309

#PassengerID,Survived - starting at 

Survived= aml.predict(x_test_h2o).as_data_frame()

print(Survived.head())

print(Survived.info())

Survived.to_csv('Survived_h2o.csv')



## Submission into kaggle

## Used regression score to change to classification -- >.5 = 1, <.5 = 0

sub = pd.DataFrame()



sub.to_csv('Titanic_h2oaml.csv', index=False)