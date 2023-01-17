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
import datetime as dt
import random
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv('../input/wecrec2020/Train_data.csv')
test_data = pd.read_csv('../input/wecrec2020/Test_data.csv')

train_data = train_data.sample(frac=1, axis=0).reset_index(drop=True)
train_data.head()
train_x = pd.DataFrame()
test_x = pd.DataFrame()

train_x['F2'] =  pd.to_datetime(train_data['F2']).map(dt.datetime.toordinal)
test_x['F2'] =  pd.to_datetime(test_data['F2']).map(dt.datetime.toordinal)

for feature in ['F3', 'F4','F5','F7', 'F8','F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15']:
    train_x[feature] = train_data[feature]
    test_x[feature] = test_data[feature]

for feature in ['F6', 'F16', 'F17']:
    
    scaler = MinMaxScaler()
    scaler.fit(train_data[feature].to_numpy().reshape(-1,1))
    train_x[feature] = scaler.transform(train_data[feature].to_numpy().reshape(-1,1))
    test_x[feature] = scaler.transform(test_data[feature].to_numpy().reshape(-1,1))

train_y = train_data['O/P']
arr_train_x = train_x.to_numpy()
arr_train_y = train_y.to_numpy()

arr_test_x = test_x.to_numpy()
# Various hyper-parameters to tune
xgb = XGBRegressor()
parameters = {'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [100, 250, 500]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose=True)

xgb_grid.fit(arr_train_x, arr_train_y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
print(test_data)
predictions = xgb_grid.best_estimator_.predict(arr_test_x)
results = pd.DataFrame()
results['id'] = test_data['Unnamed: 0']
results['PredictedValue'] = predictions
results.to_csv('result1.csv')
