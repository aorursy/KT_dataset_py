import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading the csv file using pandas

data = pd.read_csv('../input/timeseries/Train.csv')
#Checking the first  values of the  dataset

data.head()
#Checking the shape of the dataset

data.shape
#Defining the train and test in 80-20 split

train = data[0:14630]

test = data[14630:]
#First five values of the train dataset

train.head()
#First five values of test dataset

test.head()
#Function to do some processing

def processing(df):

        df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 

        df.index = df.Timestamp 

        df = df.resample('D').mean()

        df.drop('ID',1,inplace = True)

        return df
#Passing through the function

train = processing(train)

test =processing(test)

data = processing(data)
#Ploting the seasonal decompose

import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

decompose = seasonal_decompose(data, model='multiplicative')

fig = plt.figure()

fig = decompose.plot(),
#Deriving the function of Dickey fuller test for checking the stationarity

from statsmodels.tsa.stattools import adfuller

def adf_test(series):

    result = adfuller(series.dropna(),autolag='AIC')

    if result[1] <= 0.05:

        print("Data  is stationary")

    else:

        print("Data  is non-stationary")
adf_test(data["Count"])
#Seasonal difference

data["Count diff"] = data["Count"]- data["Count"].shift(12)

data["Count diff"].dropna(inplace=True)
#Checking the stationarity again

adf_test(data["Count diff"])
#Auto correlation plot and Partial correlation plot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data["Count diff"], lags= 60, alpha=0.05);

plot_pacf(data["Count diff"], lags= 60, alpha=0.05);
import seaborn as sns

import matplotlib.pyplot as plt

#Plotting data

train.Count.plot(figsize=(15,8), title= 'Daily Commuters', fontsize=14)

test.Count.plot(figsize=(15,8), title= 'Daily Commuters', fontsize=14)

plt.show()
#Code for naive method

import numpy as np

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

fit2 = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.1,optimized=False)

y_hat_avg['SES'] = fit2.forecast(len(test))

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
y_hat_avg = test.copy()

fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

plt.figure(figsize=(16,8))

plt.plot( train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')

plt.legend(loc='best')

plt.show()
#order = (p,d,q)

#p- acf plot lag

#q- pacf plot lag

#d- diffencing

y_hat_avg = test.copy()

fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2,1,4),seasonal_order=(0,1,1,7)).fit()

y_hat_avg['SARIMA'] = fit1.predict(start="2014-04-26", end="2014-09-25", dynamic=True)

plt.figure(figsize=(16,8))

plt.plot( train['Count'], label='Train')

plt.plot(test['Count'], label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.show()