# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1=pd.read_csv('../input/global-temperature-time-series.csv', delimiter=";")
from datetime import datetime as dt

def convert(x):

    return dt.strptime(x, '%Y-%d-%m')

data1['Date']=data1['Date'].transform(lambda f:convert(f))
data1=data1.drop(['Source'],axis=1)

data1.index=data1['Date']
data1=data1.drop(['Date'],axis=1)
data1=data1.sort_index()
plt.plot(data1.rolling(window='30D').mean())

plt.show()

plt.plot(data1.rolling(window='30D').std())

plt.show()
data1['Mean']=data1['Mean'].astype('float64')
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(data1['Mean'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
data=data1

data['Mean']=data1['Mean']-data1['Mean'].shift(1)

print(len(data))

data=data.dropna()

print(len(data))
plt.plot(data['Mean'].rolling(window='30D').mean())

plt.show()

plt.plot(data['Mean'].rolling(window='30D').std())

plt.show()
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(data['Mean'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(data['Mean'], nlags=20)

lag_pacf = pacf(data['Mean'], nlags=20, method='ols')

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(data['Mean'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(data['Mean'])),linestyle='--',color='gray')

plt.title('Autocorrelation Function')

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(data['Mean'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(data['Mean'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
data['Mean']

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(data['Mean'], order=(1, 1, 0))

results_ARIMA = model.fit(disp=-1)
Prediction_difference=pd.Series(results_ARIMA.fittedvalues,copy=True)
data1=data1.drop(data1.index[0])
forecast=data1['Mean']+Prediction_difference
plt.plot(data1['Mean'].rolling(window='365D').mean())

plt.plot(forecast.rolling(window='365D').mean(),color='red')

plt.show()
results_ARIMA.forecast(10)