# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
my_imputer = Imputer()
train_X = my_imputer.fit_transform(X)
from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, y, verbose=False)
param = {'n_estimators':[600],
         'learning_rate':[0.1]}
my_model = GridSearchCV(my_model, param, cv=5, scoring='neg_mean_squared_error')
my_model.fit(train_X, y, verbose=False)

param = {'learning_rate' : [0.1],
         'n_estimators' : [600],
         'max_depth' : [2]}
my_model = GradientBoostingRegressor(loss='huber', random_state=1)
my_model = GridSearchCV(my_model, param, cv=5, scoring='neg_mean_squared_error')
my_model.fit(train_X, y)


test_X = test_data.select_dtypes(exclude=['object'])
test_X = my_imputer.fit_transform(test_X)
test_preds = my_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)




# Any results you write to the current directory are saved as output.


