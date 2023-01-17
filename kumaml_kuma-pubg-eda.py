# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
srcpath = '../input/'

train = pd.read_csv(srcpath+'train_V2.csv')

test = pd.read_csv(srcpath+'test_V2.csv')
# print(train.shape)

# print(test.shape)
# train.head()
# test.head()
# train.isnull().any()
# train['winPlacePerc'] = train['winPlacePerc'].fillna(train['winPlacePerc'].median())

train['winPlacePerc'] = train['winPlacePerc'].fillna(train['winPlacePerc'].mean())
# train.isnull().any()
# test.isnull().any()
Ids = test['Id']

train.drop(['Id', 'groupId', 'matchId', 'rankPoints'], axis=1, inplace=True)

test.drop(['Id', 'groupId', 'matchId', 'rankPoints'], axis=1, inplace=True)
# train.head()
train_new = pd.get_dummies(train)

test_new = pd.get_dummies(test)

y = train_new['winPlacePerc']

X = train_new.drop(['winPlacePerc'], axis=1)
# from sklearn.neural_network import MLPRegressor
# model = MLPRegressor()

# model.fit(X, y)

# result = model.predict(test_new)

# result
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=20, max_depth=40)

model.fit(X, y)

result = model.predict(test_new)
submission = pd.DataFrame({'Id':Ids, 'winPlacePerc':result})

submission.to_csv('submission.csv', index=False)