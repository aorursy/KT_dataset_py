import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import math
from matplotlib import pyplot as plt

from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.preprocessing import *

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

import os
def process_date(x, start_time=[]):
    x = x.copy()
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in x['date']]
    SECONDS_IN_DAY = 24 * 60 * 60
    if not start_time:
        start_time.append(min(dates).timestamp() / SECONDS_IN_DAY)
    start_time = start_time[0]
    x['timestamp'] = [x.timestamp() / SECONDS_IN_DAY - start_time for x in dates]
    x['year'] = [x.year for x in dates]
    x['month'] = [x.month for x in dates]
    x['day'] = [x.day for x in dates]
    x['weekday'] = [x.weekday() for x in dates]
    x.drop('date', axis=1, inplace=True)
    return x
def process(x):
    x = process_date(x)
    x.drop('id', axis=1, inplace=True)
    return x
x_train = pd.read_csv('../input/train_data.csv', index_col='index')
y_train = pd.read_csv('../input/train_target.csv', index_col='index')
x_test = pd.read_csv('../input/test_data.csv', index_col='index')
idx = x_test.index.values
x_train.drop(y_train.index[y_train['price'] > 4000], inplace=True)
y_train.drop(y_train.index[y_train['price'] > 4000], inplace=True)
x_train = process(x_train)
x_test = process(x_test)
model = GradientBoostingRegressor(learning_rate=1e-2, n_estimators=5000, criterion='mse', max_depth=4, loss='ls').fit(x_train, y_train)
y_pred = model.predict(x_test)
my_submission = pd.DataFrame({'index': idx, 'price': y_pred})
my_submission.to_csv('./GradientBoostingRegressor1e-2_5000_noOutliers_depth4.csv', index=False)