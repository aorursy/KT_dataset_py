import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

plt.style.use('fivethirtyeight')
print(os.listdir("../input"))

import statsmodels.api as sm
from fbprophet import Prophet
import xgboost as xgb

from sklearn.metrics import mean_absolute_error
def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()

def limit(data, frm, to):
    return data[(data.index>=frm)&(data.index<to)]
energy_hourly = pd.read_csv('../input/PJME_hourly.csv', 
                            index_col=[0], parse_dates=[0])
energy_hourly.sort_index(inplace=True)

t = energy_hourly.PJME_MW.copy()
t = t.drop(t.index[t.index.duplicated()])
freq_index = pd.date_range(start=t.index[0], end=t.index[-1], freq='H')
constructed = pd.Series(index=freq_index, name='PJME_MW')
constructed.update(t)
constructed.interpolate(inplace=True)
train, test = split_data(constructed, '01-Jul-2002')

train = limit(constructed, '03-01-2011', '04-01-2011')
test  = limit(constructed, '04-01-2011', '05-01-2011')

model_A = sm.tsa.statespace.SARIMAX(constructed,
                                order=(1,1,1),
                                seasonal_order=(0,0,1,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_A = model_A.fit()
print('AIC =', results_A.aic) #AIC
#result_ARIMA = results.forecast(steps=test.shape[0])
forecast_A = results_A.forecast(steps=test.shape[0])
model_P = Prophet(interval_width=0.95)
model_P.fit(pd.DataFrame({'ds': train.index, 'y':train}))
future_dates = model_P.make_future_dataframe(periods=test.shape[0], freq='H')
results_P = model_P.predict(future_dates[train.shape[0]:])
forecast_P = results_P.set_index('ds').yhat
#model_P.make_seasonality_features(period=24*7)
def to_X(data):
    return pd.Series(data.index).apply(
        lambda x: (x - data.index[0]).components.hours) \
                                .values \
                                .reshape(-1,1)

model_X = xgb.XGBRegressor(n_estimators=30)
model_X.fit(to_X(train), train.values,
        eval_set=[(to_X(train), train.values), (to_X(test), test.values)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train

forecast_X = model_X.predict(to_X(test))
plt.figure(figsize=(16,8))
plt.plot(pd.concat([train,test]))
plt.plot(test.index, forecast_A, label='ARIMA')
plt.plot(test.index, forecast_P, label='Prophet')
plt.plot(test.index, forecast_X, label='XGBoost')
plt.legend()
mean_absolute_error(test, forecast_A), \
mean_absolute_error(test, forecast_P), \
mean_absolute_error(test, forecast_X)
