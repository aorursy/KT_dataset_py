# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb # the magic sauce (source: internet)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_V2.csv")
train.head()
train2 = pd.concat([train, pd.get_dummies(train['matchType'])], axis=1)
train2.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1, inplace=True)
train2['dumvar'] = 0 # avoid the dummy variable trap
train2.head()
train2['winPlacePerc'].isna().sum()
train2 = train2.dropna()
train2['winPlacePerc'].isna().sum()
test = pd.read_csv('../input/test_V2.csv')
test2 = pd.concat([test, pd.get_dummies(test['matchType'])], axis=1)
test2.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1, inplace=True)
test2['dumvar'] = 0
test2.head()
X, y = train2.drop(['winPlacePerc'], axis=1),train2['winPlacePerc']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)
# create the regressor
regressor = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree = 0.2, learning_rate = 0.25,
                            max_depth = 7, alpha = 0.4, n_estimators = 32)
regressor
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_valid)
# check the RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print('RMSE %f' % (rmse))
train2['winPlacePerc'].describe()
test_pred = regressor.predict(test2)
preds = pd.DataFrame(test_pred, columns=['winPlacePerc'])
test_final = pd.concat([test['Id'], preds], axis=1)
test_final.head()
test_final.to_csv('submission.csv', index = False)