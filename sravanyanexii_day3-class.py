# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

globaldata=pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
indiatemp=globaldata[globaldata.Country=='India']
indiatemp.shape
indiatemp=indiatemp[['dt', 'AverageTemperature']]
indiatemp.shape
indiatemp.head()
indiatemp.plot(kind='line')
from datetime import datetime

indiatemp.dt=pd.to_datetime(indiatemp.dt) # confirming that column is date format
indiatemp.index=indiatemp.dt  #move date to index
indiatemp.head()

indiatemp1=indiatemp.drop('dt', axis=1)
indiatemp1.head()
indiatemp1.plot(kind='line')
indiatemp1.plot(kind='line', figsize=(12, 12))
(indiatemp1.isnull().sum())
indiatemp1.describe()
indiatemp1.fillna(method='bfill', inplace=True)
(indiatemp1.isnull().sum())
indiatemp1.tail()
indiatemp1.dropna()
indiatemp1.tail()
indiatemp1.dropna(inplace=True)
indiatemp1.tail()
indiatemp2=indiatemp1['1913':]  # Selecting the data of last 100 years
indiatemp2.head()
indiatemp2.tail()
indiatemp2.plot(kind='line', figsize=(12, 12))
# Augmented Dickey Fuller Test is for Testing Stationarity of time series data
# For running any time series the data has to be stationary
# the null hypothesis of the Augmented Dickey-Fuller is that there is a unit root,
# NULL Hypo- the data is not stationary & Differenced lag to make it stationary

# the alternative that there is no unit root
# if the pvalue is less than 0.05 REJECT NULL & ACCEPT Alternative
# if the pvalue is greater than 0.05 ACCEPT or FAIL TO REJECT NULL &
#REJECT Alternate
from statsmodels.tsa.stattools import adfuller
indiatempdf=adfuller(indiatemp2.AverageTemperature)
indiatempdf
indiatemp3=indiatemp2.diff(23) #
indiatemp3.head()
indiatemp3.tail()
indiatemp3.plot(kind='line')
indiatemp3.plot(kind='line', figsize=(12,12))
indiatempdf1=adfuller(indiatemp3.AverageTemperature)
from statsmodels.tsa.seasonal import seasonal_decompose
temp2decomp=seasonal_decompose(indiatemp2, model='additive', freq=12) # freq is 52 for weeks, 12 for moths, 1 for year
temp2decomp.plot()
temp2decomp=seasonal_decompose(indiatemp2, model='multiplicative', freq=12)
temp2decomp.plot()
#temp3decomp=seasonal_decompose(indiatemp3, model='multiplicative', freq=12)
indiatemp3.head()
indiatemp3.tail()
indiatemp3.dropna(inplace=True)
temp3decomp=seasonal_decompose(indiatemp3, model='additive', freq=12)
temp3decomp.plot()
print(temp3decomp.trend)
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
plot_acf(indiatemp3.AverageTemperature, lags=50)
plot_pacf(indiatemp3.AverageTemperature, lags=50)
import itertools
p=d=q=range(0,3)
pdq=list(itertools.product(p,d,q))
from statsmodels.tsa.arima_model import ARIMA
for param in pdq:
    try:
        mod=ARIMA(indiatemp3.AverageTemperature, order=param)
        results=mod.fit()
        print ('ARIMA{} - AIC:{}'.format(param, results.aic))
    except:
        continue
              
model=ARIMA(indiatemp3.AverageTemperature, order=(2,0,2), freq='M')
arimaresult=model.fit()
arimaresult.aic
arimaresult.summary(2)
import matplotlib.pyplot as plt
plt.plot(figsize=(12,12))
plt.figure(figsize=(12,12))
plt.plot(indiatemp3.AverageTemperature)
plt.plot(arimaresult.fittedvalues, color='pink')
temppridict=arimaresult.predict('2013-01-08', '2018-01-12')
temppridict
plt.figure(figsize=(12,12))
plt.plot(indiatemp3.AverageTemperature['2008'])
plt.plot(temppridict)

temppredict2=arimaresult.forecast(60)
temppredict2
plt.plot(temppredict2)
