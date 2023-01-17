# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('../input/train.csv', low_memory=False)
raw_test_data = pd.read_csv('../input/test.csv', low_memory=False)
train_data = raw_data.drop('Survived', axis = 1)
test_data = raw_test_data.copy()
raw_data.tail().T
raw_data.describe(include='all')
raw_data.isna().sum()
train_data_age_mean = train_data.Age.mean()
train_data.Age.fillna(train_data_age_mean,inplace=True)
test_data.Age.fillna(train_data_age_mean, inplace=True)
train_data.Embarked.value_counts()
train_data.Embarked.fillna('S', inplace=True)
train_data['CabinBool'] = train_data.Cabin.isnull().astype('int64')
test_data['CabinBool'] = test_data.Cabin.isnull().astype('int64')

train_data.drop('Cabin', inplace=True, axis=1)
test_data.drop('Cabin', inplace=True, axis=1)
sex_mapping = {'male':1, 'female':0}
train_data.Sex.replace(sex_mapping, inplace=True)
test_data.Sex.replace(sex_mapping, inplace=True)
train_data.Embarked.unique()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data.Embarked.replace(embarked_mapping, inplace=True)
test_data.Embarked.replace(embarked_mapping, inplace=True)
train_data.dtypes
train_data.drop(labels = ['Name', 'PassengerId', 'Ticket'], axis=1,inplace=True)

test_data.drop(labels = ['Name', 'PassengerId', 'Ticket'], axis=1,inplace=True)
model = RandomForestRegressor(n_jobs=-1)
model.fit(train_data, raw_data.Survived)
model.score(train_data, raw_data.Survived)
results = model.predict(test_data)
test_data.Fare.fillna(test_data.Fare.mean(), inplace=True)
results = model.predict(test_data)
results = [1 if x > 0.5 else 0 for x in results]
ids = raw_test_data.PassengerId
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': results })
output.to_csv('submission.csv', index=False)