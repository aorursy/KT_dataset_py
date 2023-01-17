import gc

import os

import sys

import time

import numpy as np

import pandas as pd

import feather

import pickle



from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import log_loss, accuracy_score, mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm



from fbprophet import Prophet

from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics

from fbprophet.plot import plot_cross_validation_metric

from fbprophet.plot import add_changepoints_to_plot

from fbprophet.plot import plot_plotly

from fbprophet.plot import plot_yearly

import plotly.offline as py

from datetime import date



import warnings

warnings.filterwarnings('ignore')
os.listdir('../input/ashrae-energy-prediction')
train_ = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test_ = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

weather_train_ = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test_ = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
# select 1 building

building_id = 1085

site_id = metadata[metadata['building_id']==building_id]['site_id'].values[0]

train = train_[train_['building_id'] == building_id]

test = test_[test_['building_id'] == building_id]

train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])

train = train[['timestamp', 'meter_reading']].reset_index(drop=True)

test = test[['timestamp']].reset_index(drop=True)

train.columns = ['ds', 'y']

test.columns = ['ds']

weather_train = weather_train_[weather_train_['site_id']==site_id].reset_index(drop=True)

weather_test = weather_test_[weather_test_['site_id']==site_id].reset_index(drop=True)

weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])

weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])
train.head()
weather_train.head()
train['y'].plot();
n_test = 2000

X_train = train[:-n_test]

X_valid = train[-n_test:]

y_valid = X_valid['y']

X_valid = X_valid.drop('y', axis=1)
# holidays

# thanks to https://www.kaggle.com/rohanrao/ashrae-half-and-half

holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

            "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

            "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

            "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

            "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

            "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

            "2019-01-01"]

holidays = pd.DataFrame({

  'holiday': 'holiday',

  'ds': pd.to_datetime(holidays),

  'lower_window': 0,

  'upper_window': 0,

})

# Black Friday

holidays.loc[holidays['ds'] == pd.to_datetime('2016-11-24'), 'upper_window'] = 1

holidays.loc[holidays['ds'] == pd.to_datetime('2017-11-23'), 'upper_window'] = 1

holidays.loc[holidays['ds'] == pd.to_datetime('2018-11-22'), 'upper_window'] = 1
params_prophet = {

    'growth': 'logistic', # Prophet allows you to make forecasts using a logistic growth trend model

    'changepoint_prior_scale' :0.03, # trend flexibility default 0.05

    'holidays': holidays

}
X_train['floor'] = X_train['y'].min()

X_train['cap'] = X_train['y'].max()
X_train
m = Prophet(**params_prophet)

m.fit(X_train)
#future = m.make_future_dataframe(periods=365*24, freq='H')

future = pd.concat([X_train[['ds']], X_valid])

future['floor'] = X_train['y'].min()

future['cap'] = X_train['y'].max()

fcst = m.predict(future)

fig = m.plot(fcst)
fig = m.plot(fcst)

a = add_changepoints_to_plot(fig.gca(), m, fcst)
fig = m.plot_components(fcst)
weather_train = weather_train.fillna(0)

X_train = pd.merge(X_train, weather_train.drop('site_id', axis=1), left_on = 'ds', right_on = 'timestamp').drop('timestamp', axis=1)

future = pd.merge(future, weather_train.drop('site_id', axis=1), left_on = 'ds', right_on = 'timestamp').drop('timestamp', axis=1)

X_train.head()
cols = ['air_temperature', 'cloud_coverage','dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']

m = Prophet(**params_prophet)

for col in cols:

    m.add_regressor(col)

m.fit(X_train)
fcst = m.predict(future)

fig = m.plot(fcst)
weather_test = weather_test.fillna(0)

test['floor'] = X_train['y'].min()

test['cap'] = X_train['y'].max()

X_test = pd.merge(test, weather_test.drop('site_id', axis=1), left_on = 'ds', right_on = 'timestamp').drop('timestamp', axis=1)

X_test.head()
fcst = m.predict(X_test)

preds = fcst.yhat
preds.plot();