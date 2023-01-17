#### General Import



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from datetime import datetime

import datetime

import plotly.graph_objects as go

import plotly.express as px

import folium

from folium import plugins

import warnings

import seaborn as sns

plt.style.use('ggplot')



import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm



from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

# TSA from Statsmodels

import statsmodels.api as sm

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt



from sklearn.metrics import mean_squared_error



def fxn():

    warnings.warn("deprecated", DeprecationWarning)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    fxn()

    

#### Load the data



df1 = pd.read_csv('/kaggle/input/eda-sensors/df1.csv')

df2 = pd.read_csv('/kaggle/input/eda-sensors/df2.csv')

df3 = pd.read_csv('/kaggle/input/eda-sensors/df3.csv')

df4 = pd.read_csv('/kaggle/input/eda-sensors/df4.csv')

df5 = pd.read_csv('/kaggle/input/eda-sensors/df5.csv')





#### Resample the data



# lets create time series from weather 

timeSeries = df1.loc[:, ["datetime","liv"]]

timeSeries['datetime'] = pd.to_datetime(timeSeries['datetime'] )

ts = timeSeries.set_index('datetime').resample('1H').max().reset_index()

from pandas import read_csv

from matplotlib import pyplot

# adfuller library 

from statsmodels.tsa.stattools import adfuller

#  kpss library

from statsmodels.tsa.stattools import kpss



def summary_statistics(series):

    X = series.values

    split = round(len(X) / 2)

    X1, X2 = X[0:split], X[split:]

    mean1, mean2 = X1.mean(), X2.mean()

    var1, var2 = X1.var(), X2.var()

    print('Summary Statistics')

    print('mean1=%f, mean2=%f' % (mean1, mean2))

    print('variance1=%f, variance2=%f' % (var1, var2))

    print('')





# check_adfuller

def check_adfuller(series):

    # Dickey-Fuller test

    print ('Results of adfuller Test:')

    result = adfuller(series, autolag='AIC')

    print('Test statistic: ' , result[0])

    print('p-value: '  ,result[1])

    print('Critical Values:' ,result[4])

    print('')

    



#define KPSS

def check_kpss(series):

    print ('Results of KPSS Test:')

    result = kpss(series, regression='c', nlags='auto')

    print('Test statistic: ' , result[0])

    print('p-value: '  ,result[1])

    print('Critical Values:' ,result[3])

    print('')





# check_mean_std

def check_mean_std(series):

    #Rolling statistics

    TS = series

    TS['rollmean'] = TS.liv.rolling(12).mean()

    TS['rollstd'] = TS.liv.rolling(12).std()



    # Create traces

    fig = go.Figure()

    fig.add_trace(go.Scatter(x = TS['datetime'], y=TS['liv'], name='Original'))

    fig.add_trace(go.Scatter(x = TS['datetime'], y=TS['rollmean'], name='Rollling Mean'))

    fig.add_trace(go.Scatter(x = TS['datetime'], y=TS['rollstd'], name='Rolling Std'))

    fig.update_layout(title='Check Stationarity with Rolling Mean and Rolling Std ',xaxis_title='Datetime')

    fig.show()

    TS.drop(['rollmean','rollstd'],axis=1,inplace=True)



# Examine the patterns of ACF and PACF (along with the time series plot and histogram)



def tsplot(ts, lags=None, title='', figsize=(14, 8)):

    '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.

    '''

    y = ts.liv

    fig = plt.figure(figsize=figsize)

    layout = (2, 2)

    ts_ax   = plt.subplot2grid(layout, (0, 0))

    hist_ax = plt.subplot2grid(layout, (0, 1))

    acf_ax  = plt.subplot2grid(layout, (1, 0))

    pacf_ax = plt.subplot2grid(layout, (1, 1))

    

    y.plot(ax=ts_ax)

    ts_ax.set_title(title)

    y.plot(ax=hist_ax, kind='hist', bins=25)

    hist_ax.set_title('Histogram')

    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]

    sns.despine()

    fig.tight_layout()

    

    summary_statistics(y)

    check_adfuller(y)

    check_kpss(y)

    check_mean_std(ts)

    

    return ts_ax, acf_ax, pacf_ax

tsplot(ts)
ts_diff_first = timeSeries.set_index('datetime').resample('1H').max().reset_index()

# 1st order differencing

ts_diff_first.liv = ts_diff_first.liv.diff()

ts_diff_first.dropna(inplace=True)

# visualization

tsplot(ts_diff_first)
ts_diff_seas = timeSeries.set_index('datetime').resample('1H').max().reset_index()

# Seasonal differencing

ts_diff_seas.liv = ts_diff_seas.liv.diff(24)

ts_diff_seas.dropna(inplace=True)

# visualization

tsplot(ts_diff_seas)
# create a n-th order differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return diff



 

# invert differenced forecast

def inverse_difference(last_ob, value):

    return value + last_ob
TS = ts.liv

v_0 = TS[:24]

diff_24 = difference(TS,24)
TRAIN = TS[:-24]

TEST = TS[-24:]



train = diff_24[:-24]

test = diff_24[-24:] #predictions last day
from statsmodels.tsa.holtwinters import ExponentialSmoothing



# fit model

model_es = ExponentialSmoothing(train, seasonal="add", seasonal_periods=24)

model_es_fit = model_es.fit()



# forecast

pred_es = model_es_fit.forecast(24)



# stack with train dataset

diff_predicted_es = np.hstack((train, pred_es))

diff_target = np.hstack((train,test))





# total invert differencing

predicted_es =  np.hstack((v_0,[inverse_difference(TRAIN[i], diff_predicted_es[i]) for i in range(len(diff_predicted_es))]))

target =  np.hstack((v_0,[inverse_difference(TRAIN[i], diff_target[i]) for i in range(len(diff_target))]))



y_pred = predicted_es[-24:]

y_true = target[-24:]



# MSE

mse = mean_squared_error(y_true, y_pred)

print('MSE')

print(mse)





# plot

plt.plot(target[2150:])

plt.plot(predicted_es[2150:])
pyplot.figure()

smt.graphics.plot_acf(ts_diff_seas.liv)

smt.graphics.plot_pacf(ts_diff_seas.liv)

pyplot.show()
from statsmodels.tsa.arima_model import ARIMA



# fit model

model_arima = ARIMA(train, order=(6,0,3))

model_arima_fit = model_arima.fit(disp=0)



# forecast

pred_arima = model_arima_fit.forecast(steps=24)[0]



# stack

diff_predicted_arima = np.hstack((train, pred_arima))



#invert

predicted_arima =  np.hstack((v_0,[inverse_difference(TRAIN[i], diff_predicted_arima[i]) for i in range(len(diff_predicted_arima))]))





y_pred_arima = predicted_arima[-24:]



# MSE

mse = mean_squared_error(y_true, y_pred_arima)

print('MSE')

print(mse)



# plot

plt.plot(target[2150:])

plt.plot(predicted_arima[2150:])

model_sarima = sm.tsa.statespace.SARIMAX(train,order=(6,0,1), seasonal_order=(4,1,1,12))#, simple_differencing=True) 



model_sarima_fit = model_sarima.fit(disp=False)



# forecast

pred_sarima = model_sarima_fit.forecast(steps=24)



# stack

diff_predicted_sarima = np.hstack((train, pred_sarima))



# invert 

predicted_sarima =  np.hstack((v_0,[inverse_difference(TRAIN[i], diff_predicted_sarima[i]) for i in range(len(diff_predicted_sarima))]))





y_pred_sarima = predicted_sarima[-24:]



# MSE

mse = mean_squared_error(y_true, y_pred_sarima)

print('MSE')

print(mse)



# plot

plt.plot(target[2150:])

plt.plot(predicted_sarima[2150:])