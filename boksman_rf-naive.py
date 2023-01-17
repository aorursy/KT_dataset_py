# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
pd.set_option('display.precision',5)
data = pd.read_csv('../input/train_V2.csv')
print(data.shape)
data.head(2)
groupInfo = pd.DataFrame(data[['Id', 'groupId']].groupby('groupId')['Id'].size()).reset_index()
groupInfo.rename({'Id': 'groupSize'}, axis=1, inplace=True)
print(groupInfo.shape)
groupInfo.head(2)
data = pd.merge(data, groupInfo, on='groupId', how='left')
data.dropna(subset=['matchType', 'winPlacePerc'], inplace=True)
print(data.shape)
data.head(2)
data = data[data['groupSize'] <= 4]
data.dropna(inplace=True)
print(data.shape)
data.head(2)
y = data['winPlacePerc']
y.shape
x = data.copy()
x.drop(['winPlacePerc', 'matchType', 'Id', 'groupId', 'matchId'], axis=1, inplace=True)
print(x.shape)
x.head(2)
test_data = pd.read_csv('../input/test_V2.csv')
x_test = test_data.copy()
print(x_test.shape)
x_test.head(2)
groupInfo = pd.DataFrame(x_test[['Id', 'groupId']].groupby('groupId')['Id'].size()).reset_index()
groupInfo.rename({'Id': 'groupSize'}, axis=1, inplace=True)
print(groupInfo.shape)
groupInfo.head(2)
x_test = pd.merge(x_test, groupInfo, on='groupId', how='left')
#X_test.dropna(subset=['matchType', 'winPlacePerc'], inplace=True)
print(x_test.shape)
x_test.head(2)
x_test.drop(['matchType', 'Id', 'groupId', 'matchId'], axis=1, inplace=True)
print(x_test.shape)
x_test.head(2)
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
rf = RandomForestRegressor(max_depth=50, max_features=15, n_estimators=30, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_val)
print(mean_absolute_error(y_pred, y_val))
y_test = rf.predict(x_test)
result = pd.DataFrame(test_data['Id']).join(pd.DataFrame(pd.Series(y_test, name='winPlacePerc')))
result.set_index('Id', drop=True, inplace=True)
print(result.shape)
result.head(2)
result.to_csv('rf_naive_submission.csv')
result.head(2)
print(os.listdir())
