# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

from math import sqrt

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import the csv file

oilPrices = pd.read_csv('/kaggle/input/brent-oil-prices/BrentOilPrices.csv')

#change column names to more comfortable names

oilPrices.columns=['date', 'price']



print("Data Set:"% oilPrices.columns, oilPrices.shape)

print("Data Types:", oilPrices.dtypes)

#Check the top five records

oilPrices.head()
#Cast Date Column to type date

oilPrices['date'] = pd.to_datetime(oilPrices['date'])
#As you may noticed the time series data does not contain the values for Saturday and Sunday. Hence missing values have to be filled. 

#Fill in Weekends - First make date as index (for resample method), then use forward fill ffill(),

#which will assign the weekend values with Friday value. Resample method for frequency conversion and resampling of time series. Object must have a datetime-like index (DatetimeIndex, PeriodIndex, or TimedeltaIndex), 

#or pass datetime-like values to the on or level keyword

oilPrices.set_index('date', inplace=True)

oilPrices = oilPrices.resample('D').ffill().reset_index()
#Make sure we have no null values

oilPrices.isnull().values.any()
#Let us split the date into year, month and week to explore trend in oil prices

oilPrices['year']=oilPrices['date'].dt.year

oilPrices['month']=oilPrices['date'].dt.month

oilPrices['week']=oilPrices['date'].dt.week
#Let us read the data until the 1st of January 2019 to split the data and predict prices in 2019

train = oilPrices[(oilPrices['date' ] > '2001-01-01') & (oilPrices['date' ] <= '2019-12-31')]

test = oilPrices[oilPrices['date' ] >= '2020-01-01']
#Yearly price visualization

yearlyPrice=train.groupby(["year"])['price'].mean()

plt.figure(figsize=(16,4))

plt.title('Oil Prices')

plt.xlabel('Year')

plt.ylabel('Price')

yearlyPrice.plot()

plt.show();
#time-series to decompose our time series into three distinct components: trend, seasonality, and noise.

monthlyPrice=oilPrices.groupby(["month"])['price'].mean()

from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(yearlyPrice, freq=1, model='additive')

fig = decomposition.plot()

plt.show()
from fbprophet import Prophet

d={'ds':train['date'],'y':train['price']}

df_pred=pd.DataFrame(data=d)

# I took off Seasonality as Oil prices on weekends remain same as Friday until next opening on Monday

model = Prophet(daily_seasonality=False)

model.fit(df_pred)

future = model.make_future_dataframe(periods=273)

forecast = model.predict(future)

forecast2020 = forecast[(forecast['ds' ] >= '2020-01-01') & (forecast['ds' ] <= '2020-04-21')]
plt.figure(figsize=(18, 6))

model.plot(forecast, xlabel = 'Date', ylabel = 'Price')

plt.title('Brent Oil Price Prediction');
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(model, forecast2020)  # This returns a plotly Figure

py.iplot(fig)
# Create plots with pre-defined labels.

fig, ax = plt.subplots()

ax.plot(forecast2020['ds'], forecast2020['yhat'], label='Predicted Prices')

ax.plot(test['date'], test['price'], label='Original Prices')

plt.ylim([0,100])

legend = ax.legend(loc='upper center', shadow=True)

plt.title('Prophet Model Brent Oil Prices Forecast 2020')

plt.xlabel('Month')

plt.ylabel('Price')

plt.show()
import statistics

#Create a series of predicted values and observed ones

observed=test['price'].iloc[1:]

predicted=forecast2020['yhat'].iloc[1:]

#Reset the index of the series

predicted.reset_index(drop=True, inplace=True)

observed.reset_index(drop=True, inplace=True)

# loop over the set and find the difference between observed and predicted values then save them in a set

pred_err=[]

for count in range(len(observed)):

    err = predicted[count] - observed[count]

    pred_err.append(err)

#Take the Absolute value and find the mean

mae = statistics.mean(map(abs,pred_err))

print('Mean Absolute Error = {}'.format(round(mae, 2)))
#Convert to Time Series For ARIMA Estimator

series=pd.Series(data=train['price'].to_numpy(), index=train['date'])

#check if the Index is Datetime format

series.index
#The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.

from statsmodels.tsa.stattools import adfuller

from numpy import log

result = adfuller(series)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
#Look if there is a stationary data, which looks non stationary

#We need stationary data to make time series forecasting

plt.plot(series[0:100])

plt.show()
#find the order of differencing (d) in ARIMA model; hence the purpose of differencing it to make the time series stationary

daily_series_diff1 = series.diff(periods=1).dropna()

daily_series_diff2 = daily_series_diff1.diff(periods=1).dropna()

fig, ax = plt.subplots()

ax.plot(daily_series_diff1[0:100], label='1st Order Differencing')

ax.plot(daily_series_diff2[0:100], label='2nd Order Differencing')

plt.ylim([-3,3])

legend = ax.legend(loc='upper center', shadow=True)

plt.title('Time Series')

plt.xlabel('Date')

plt.ylabel('Diff')

plt.show()
plt.rcParams.update({'figure.figsize':(12,3), 'figure.dpi':120})

from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 2, sharex=True)

plot_acf(daily_series_diff1, lags=20, ax=axes[0], title="Autocorrelation 1st Order Differencing")

plot_acf(daily_series_diff2, lags=20, ax=axes[1], title="Autocorrelation 2nd Order Differencing")

plt.xlabel('Lag')

plt.ylabel('ACF')

plt.show()
#Determine the number of the moving average by looking at the Partial Autocorrelation : p value should be one based on the Partial Autocorrelation 

plt.rcParams.update({'figure.figsize':(12,3), 'figure.dpi':120})

#Partial Auto-Correlation

from statsmodels.graphics.tsaplots import plot_pacf

fig, axes = plt.subplots(1, 2, sharex=True)

plot_pacf(daily_series_diff1, lags=10, ax=axes[0], title="Partial Autocorrelation 1st Order Differencing")

plot_pacf(daily_series_diff2, lags=10, ax=axes[1], title="Partial Autocorrelation 2nd Order Differencing")

plt.xlabel('Lag')

plt.ylabel('PACF')

plt.show()
!pip install pmdarima

#Number of differences required for a stationary series

from pmdarima.arima.utils import ndiffs

y=series

# augmented Dickey–Fuller test (adf test)

print("ADF Test: ",ndiffs(y, test='adf'))

# Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test

print("KPSS Test: ",ndiffs(y, test='kpss'))

# Phillips–Perron (PP) test:

print("PP Test: ",ndiffs(y, test='pp'))
import pmdarima as pm

model = pm.auto_arima(series, start_p=1, start_q=1,

                      test='adf',       # use adftest to find optimal 'd'

                      max_p=3, max_q=3, # maximum p and q

                      m=1,              # frequency of series

                      d=None,           # let model determine 'd'

                      seasonal=False,   # No Seasonality

                      start_P=0, 

                      D=0, 

                      trace=True,

                      error_action='ignore',  

                      suppress_warnings=True, 

                      stepwise=True)



print(model.summary())
from statsmodels.tsa.arima_model import ARIMA

# fit model

model = ARIMA(series, order=(1, 0, 1)).fit(transparams=False)

print(model.summary())
#Forecast the oil prices for the period start='1/1/2019', end='9/30/2019'

#typ='levels' if d is not set to zero (d = the number of nonseasonal differences)

ARIMA_Predict = model.predict(start='1/1/2019', end='9/30/2019')
#Standard deviation of residuals or Root-mean-square error (RMSD) https://www.youtube.com/watch?v=zMFdb__sUpw

mseProphet = mean_squared_error(test['price'],forecast2020['yhat'])

mseARIMA = mean_squared_error(test['price'],ARIMA_Predict)

rmseProphet = sqrt(mseProphet)

rmseARIMA = sqrt(mseARIMA)

print('The Mean Squared Error of ARIMA forecasts is {}'.format(round(mseARIMA, 2)))

print('The Root Mean Squared Error of ARIMA forecasts is {}'.format(round(rmseARIMA, 2)))

print('The Mean Squared Error of Prophet forecasts is {}'.format(round(mseProphet, 2)))

print('The Root Mean Squared Error of Prophet forecasts is {}'.format(round(rmseProphet, 2)))
#OR you may replace all the above with sklearn simple mae function:

from sklearn.metrics import mean_absolute_error

maeARIMA=mean_absolute_error(test['price'],ARIMA_Predict)

maeProphet=mean_absolute_error(test['price'],forecast2020['yhat'])

print('Mean Absolute Error ARIMA = {}'.format(round(maeARIMA, 2)))

print('Mean Absolute Error Prophet = {}'.format(round(maeProphet, 2)))
# Create plots with pre-defined labels.

fig, ax = plt.subplots()

ax.plot(forecast2020['ds'], ARIMA_Predict, label='Predicted Prices')

ax.plot(test['date'], test['price'], label='Original Prices')

plt.ylim([0,100])

legend = ax.legend(loc='upper center', shadow=True)

plt.title('ARIMA Model Brent Oil Prices Forecast 2019')

plt.xlabel('Month')

plt.ylabel('Price')

plt.show()
mae=mean_absolute_error(test['price'],ARIMA_Predict)

print('Mean Absolute Error = {}'.format(round(mae, 2)))