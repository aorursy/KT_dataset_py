# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', parse_dates=['date'])
train.head()
train['y'] = train['date'].dt.year
train['m'] = train['date'].dt.month
train['d'] = train['date'].dt.dayofweek

train.drop('date', axis=1, inplace=True)

sales = train.pop('sales')
test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col='id')

test['y'] = test['date'].dt.year
test['m'] = test['date'].dt.month
test['d'] = test['date'].dt.dayofweek

test.drop('date', axis=1, inplace=True)
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(train.values, sales.values)
y_pred = model.predict(test.values)
submission = pd.read_csv('../input/sample_submission.csv', index_col='id')
submission['sales'] = y_pred
submission.to_csv('simple_rf_benchmark.csv')
