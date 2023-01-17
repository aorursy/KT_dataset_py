# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt

import warnings  

import statsmodels.api as sm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/timeseries/Train.csv')

test = pd.read_csv('/kaggle/input/timeseries/Test.csv')
train
train.drop('ID',axis = 1,inplace = True)
train
from statsmodels.tsa.stattools import adfuller

result = adfuller(train['Count'])

print(result)
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train['Datetime']
test.index = test['Datetime']
train
train.drop('Datetime',1,inplace = True)
plt.figure(figsize=(15,7))

plt.plot(train['Count'])

plt.show()
rolling_mean = train['Count'].rolling(24).mean()

plt.figure(figsize=(12,8))

plt.plot(train['Count'],color = 'red')

plt.plot(rolling_mean,color = 'blue')

plt.plot()
from statsmodels.graphics.tsaplots import acf, pacf

plt.plot(acf(train['Count'],nlags = 49))
from statsmodels.tsa.seasonal import seasonal_decompose 

decomposition = seasonal_decompose(train['Count'].diff().dropna()) 

decomposition.plot()
train_rol = train.rolling(24).mean()
train_rol
plt.figure(figsize=(15,7))

plt.plot(train_rol['Count'].dropna())

plt.show()
result = adfuller(train['Count'].diff().dropna())

print(result)
from statsmodels.graphics.tsaplots import acf,pacf

plt.plot(acf(train['Count'].diff().dropna(),nlags = 10))

plt.plot(pacf(train['Count'].diff().dropna(),nlags = 10))

plt.show()
Train=train.ix['2012-08-25':'2014-06-24'] 

valid=train.ix['2014-06-25':'2014-09-25']
train.shape
Train
Train.shape
valid
valid.shape
import statsmodels.api as sm

y_hat_avg = valid.copy() 

fit2 = sm.tsa.statespace.SARIMAX(Train.Count, order=(2, 1, 4),seasonal_order=(1,1,1,24)).fit() 

y_hat_avg['SARIMA'] = fit2.predict(start="2014-6-25", end="2014-9-26", dynamic=True) 

plt.figure(figsize=(16,8)) 

plt.plot( Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 

plt.legend(loc='best') 

plt.show()
from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(valid, y_hat_avg.SARIMA.dropna()))

print(rms)
plt.figure(figsize=(16,8)) 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 

plt.legend(loc='best') 

plt.show()
submission = test.copy()
submission['Count'] = fit2.predict(start="2014-9-26", end="2015-4-27", dynamic=True)

submission.drop(['Datetime'],axis = 1,inplace = True)
submission
submission.to_csv("Time Series.csv",index=False)
train
daily = train.Count.resample('D').mean()

monthly = train.Count.resample('M').mean()

yearly = train.Count.resample('Y').mean()
plt.figure(figsize=(12,8))

plt.plot(train['Count'])

plt.plot(daily)

plt.plot(monthly)

plt.plot(yearly)

plt.show()
Traind=daily.ix['2012-08-25':'2014-06-24'] 

validd=daily.ix['2014-06-25':'2014-09-25']
result = adfuller(daily.diff().dropna())

print(result)
plt.plot(acf(daily,nlags = 14))
plt.plot(pacf(daily))
plt.figure(figsize=(20,4))

decomposition = seasonal_decompose(daily.diff().dropna()) 

decomposition.plot()

plt.show()

import statsmodels.api as sm

y_hat_avgd = validd.copy() 

fit1 = sm.tsa.statespace.SARIMAX(Traind, order=(2, 1, 4),seasonal_order=(2,1,2,7)).fit() 

y_hat_avgd['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True) 

plt.figure(figsize=(16,8)) 

plt.plot( Traind, label='Train') 

plt.plot(validd, label='Valid') 

plt.plot(y_hat_avgd['SARIMA'], label='SARIMA') 

plt.legend(loc='best') 

plt.show()
validd.shape
y_hat_avgd.SARIMA.shape
from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(validd, y_hat_avgd.SARIMA))

print(rms)
fit1.plot_diagnostics()

plt.show()