import os

import re

import json

import pandas as pd

from pandas import to_datetime

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from collections import OrderedDict

import numpy as np



train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



map_Embarked = {'C':0,'Q':1,'S':2}

train_data['Embarked'] = train_data['Embarked'].map(map_Embarked)

test_data['Embarked'] = test_data['Embarked'].map(map_Embarked)



map_Sex = {'male':0,'female':1}

train_data['Sex'] = train_data['Sex'].map(map_Sex)

test_data['Sex'] = test_data['Sex'].map(map_Sex)



test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



train_data.dropna(subset=['Pclass','Sex','Age','Fare','Embarked'],inplace=True)



X_train, X_test, y_train, y_test = train_test_split(train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']],

                                                        train_data['Survived'], test_size=0.1, random_state=0)

params = {'n_estimators':500, 'max_depth':4, 'min_samples_split': 2,

              'learning_rate': 0.01, 'loss': 'ls'}

clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

mse = mean_squared_error(y_test, clf.predict(X_test))

print("MSE: %.4f" % mse)



test_data['Survived'] = clf.predict(test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])

result = test_data[['PassengerId','Survived']]

result['Survived'] = result['Survived'].apply(lambda x:round(x))

result.to_csv('submission.csv',index=False)