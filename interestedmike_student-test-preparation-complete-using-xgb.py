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
students_data = pd.read_csv('../input/StudentsPerformance.csv')
students_data.head
students_data.describe()
students_data.columns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

y = students_data['test preparation course']
LEncoder = LabelEncoder()
LEncoder.fit(y) # fit labels
y = LEncoder.transform(y) # transform the labels
y = pd.get_dummies(y)

feature_columns = ['math score', 'reading score', 'writing score']
X = students_data[feature_columns]
MNScaler = MinMaxScaler()
MNScaler.fit(X) # fit math, reading, and writing scores
X = MNScaler.transform(X) # transform the scores

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Using Random Forest
random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error

preds = random_forest_model.predict(val_X)
mae = mean_absolute_error(val_y, preds)
print("Mean absolute error is: {:,.5f}".format(mae))
print(random_forest_model.score(train_X, train_y))
print(random_forest_model.score(val_X, val_y))
# Using XGBRegressor
from xgboost import XGBRegressor 

train_X, test_X, train_y, test_y = train_test_split(X, y.as_matrix(), random_state=1)
xgb_model = XGBRegressor()
xgb_model.fit(train_X, train_y, verbose=False)
predictions = xgb_model.predict(test_X)
predictions