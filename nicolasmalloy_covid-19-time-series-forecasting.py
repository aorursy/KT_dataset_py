# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import io
import requests
import seaborn as sns
import time
import datetime
from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import linear_model
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv", header=0, index_col=0)
test_data = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", header=0, index_col=0)
submission_data = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv", encoding= 'unicode_escape')
#Grouped according to date Globally
grouped_train_data = train_data.groupby(train_data['Date']).sum().reset_index()
grouped_train_data.head()
grouped_test_data = test_data.groupby(test_data['Date']).sum().reset_index()
grouped_test_data = grouped_test_data.drop('Country_Region',axis=1)
grouped_test_data.head()
from datetime import datetime
con=grouped_train_data['Date']
grouped_train_data['Date']=pd.to_datetime(grouped_train_data['Date'])
grouped_train_data.set_index('Date', inplace=True)
#check datatype of index
grouped_train_data.index
from datetime import datetime
con=grouped_test_data['Date']
grouped_test_data['Date']=pd.to_datetime(grouped_test_data['Date'])
grouped_test_data.set_index('Date', inplace=True)
#check datatype of index
grouped_test_data.index
#convert to time series:
ts = grouped_train_data['Fatalities']
ts.head(10)
ts.plot()
plt.show()
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(12).mean()
    #rolstd = pd.rolling_std(timeseries, window=12)#Plot rolling statistics:
    rolstd = timeseries.rolling(12).std()
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xticks(rotation=90)
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
test_stationarity(ts)
ts_log = np.log(ts)
plt.plot(ts_log)
plt.xticks(rotation=90)
plt.show()
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg,color='red')
plt.xticks(rotation=90)
plt.show()
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head()
ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head()
test_stationarity(ts_log_moving_avg_diff)
expwighted_avg = ts_log.ewm(12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg,color='red')
plt.xticks(rotation=90)
plt.show()
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)
#First Take Difference
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.xticks(rotation=90)
plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(15, 9))
plt.subplot(411)
plt.plot(ts_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_decompose)
from statsmodels.tsa.arima_model import ARIMA
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
model = ARIMA(ts_log, order=(3,1,0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.xticks(rotation=90)
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
plt.show()
model = ARIMA(ts_log, order=(0,1,8))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.xticks(rotation=90)
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
plt.show()
model = ARIMA(ts_log, order=(3,1,8))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.xticks(rotation=90)
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.xticks(rotation=90)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
fig, ax = plt.subplots()
ax = ts_log.loc['2020-01-22':].plot(ax=ax,figsize=(10,5))
fig = results_ARIMA.plot_predict('2020-04-14', '2020-8-15', dynamic=True,ax=ax,plot_insample=False)
plt.legend(loc='upper left')
plt.grid()
plt.show()
fig, ax = plt.subplots()
ax = ts_log.loc['2020-01-22':].plot(ax=ax,figsize=(10,5))
fig = results_AR.plot_predict('2020-04-14', '2020-8-15', dynamic=True,ax=ax,plot_insample=False)
plt.legend(loc='upper left')
#plt.xticks(rotation=90)
plt.grid()
plt.show()
fig, ax = plt.subplots()
ax = ts_log.loc['2020-01-22':].plot(ax=ax,figsize=(10,5))
fig = results_MA.plot_predict('2020-04-14', '2020-8-15', dynamic=True,ax=ax,plot_insample=False)
plt.legend(loc='upper left')
plt.grid()
plt.show()
