# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

df = pd.read_csv('../input/timeseries/Train.csv')

df.head()
df.tail()
#Subsetting the dataset

#Index 11856 marks the end of year 2013

df = pd.read_csv('../input/timeseries/Train.csv', nrows = 11856)



#Creating train and test set 

#Index 10392 marks the end of October 2013 

train=df[0:10392] 

test=df[10392:]
#Aggregating the dataset at daily level

df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 

df.index = df.Timestamp 

df = df.resample('D').mean()

train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

train = train.resample('D').mean() 

test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

test.index = test.Timestamp 

test = test.resample('D').mean()
train.head()
#Plotting data

train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)

plt.show()
dd= np.asarray(train.Count)

y_hat = test.copy()

y_hat['naive'] = dd[len(dd)-1]

plt.figure(figsize=(12,8))

plt.plot(train.index, train['Count'], label='Train')

plt.plot(test.index,test['Count'], label='Test')

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')

plt.legend(loc='best')

plt.title("Naive Forecast")

plt.show()
from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(test.Count, y_hat.naive))

print(rms)
y_hat_avg = test.copy()

y_hat_avg['avg_forecast'] = train['Count'].mean()

plt.figure(figsize=(12,8))

plt.plot(train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.avg_forecast))

print(rms)
y_hat_avg = test.copy()

y_hat_avg['moving_avg_forecast'] = train['Count'].rolling(60).mean().iloc[-1]

plt.figure(figsize=(16,8))

plt.plot(train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.moving_avg_forecast))

print(rms)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

y_hat_avg = test.copy()

fit2 = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6,optimized=False)

y_hat_avg['SES'] = fit2.forecast(len(test)) # Simple Exponential Smoothing.

plt.figure(figsize=(16,8))

plt.plot(train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['SES'], label='SES')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.SES))

print(rms)
import statsmodels.api as sm

sm.tsa.seasonal_decompose(train.Count).plot()

result = sm.tsa.stattools.adfuller(train.Count)

plt.show()
y_hat_avg = test.copy()



fit1 = Holt(np.asarray(train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

y_hat_avg['Holt_linear'] = fit1.forecast(len(test))



plt.figure(figsize=(16,8))

plt.plot(train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_linear))

print(rms)
y_hat_avg = test.copy()

fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

plt.figure(figsize=(16,8))

plt.plot( train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_Winter))

print(rms)
y_hat_avg = test.copy()

fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()

y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)

plt.figure(figsize=(16,8))

plt.plot( train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat_avg.SARIMA))

print(rms)