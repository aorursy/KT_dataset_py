# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pmdarima
# Function to read data into Kernel

ts = pd.read_csv("../input/" + 'nab/realAdExchange/realAdExchange/exchange-2_cpc_results' + ".csv")
ts.head()
ts['timestamp']= pd.to_datetime(ts['timestamp']) #convert the col to timestamo data
ts.set_index('timestamp',drop=True,inplace=True) #set it as the index
ts=ts.resample('h').mean() #sampling
ts.columns=['cpc'] #set the column name as cpc
ts=ts['cpc']
ts.head()
ts.isna().sum()
# 25 observations are missing 

ts.index[ts.isna()] #printng out the missing value location
# we see that most of the missing values are on 02.09.2011
ts.plot(grid=True,figsize =(19,8))
# in this plot we can see the missing values

ts_clean = ts.loc[ts.index < '2011-09-02'] # selecting the missing values
ts_clean.fillna(method='pad', inplace=True)  # this is equivalent to both method='ffill' and .ffill()
ts_clean.head()
ts_clean.plot(grid=True, figsize=(19,8)) # cleaned data
# in practise 70-15-15 is the best way to split the data into Test-Train-Final test(not used for CV or training)
n_obs = ts_clean.shape[0]  #shape -1512
split1= ts_clean.index[int(0.7*n_obs)]  # split based on index
split2= ts_clean.index[int(0.85*n_obs)]  # split based on index

train_ts = ts_clean.loc[ts_clean.index <= split1]
val_ts = ts_clean.loc[(ts_clean.index > split1) & (ts_clean.index <= split2)]
test_ts = ts_clean.loc[ts_clean.index > split2]
pd.concat([train_ts.rename('Train'),test_ts.rename('Test'),val_ts.rename('Validation')],axis=1).plot(grid=True,figsize =(18,9),legend =True)
from statsmodels.tsa.seasonal import seasonal_decompose


decomposition = seasonal_decompose(train_ts,model='additive',freq =24) # frequency of 24hrs -- hourly data set
plt.figure(figsize=(18,9))
plt.plot(decomposition.trend)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualize 
plt.figure(figsize=(18,9))
plt.subplot(411)
plt.plot(train_ts, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

from statsmodels.tsa.stattools import adfuller

X = train_ts.values
adf_result = adfuller(X,regression="c")
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])

print('Critical Values: The ‘1%’, ‘10%’ and ‘5%’ values are the critical values for 99%, 90% and 95% confidence levels')
for key, value in adf_result[4].items():
    print('\t%s: %.3f' % (key, value))
    
if adf_result[0] < adf_result[4]['5%']:
    print('H0 hypothesis rejected : Stationary time series')
else:
    print('H1 hypothesis accepted : Non-Stationary time series')
resid = residual.dropna()

adf_resid = adfuller(resid,regression="c")
print ('ADF Statistic is :', adf_resid[0])
print ('p-value is :', adf_resid[1])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

resid = residual.dropna()

plt.figure(figsize=(18,9))
plt.subplot(211)
plot_acf(resid, lags=52, ax=plt.gca())
plt.grid()
plt.subplot(212)
plot_pacf(resid, lags=52, ax=plt.gca())
plt.grid()
plt.show()
import pmdarima as pm
stepwise_model = pm.auto_arima(train_ts, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=24,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
arima = pm.auto_arima(train_ts, error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=50,
                      seasonal=True, m=24)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time

model = SARIMAX(train_ts, order=(3, 1, 3), seasonal_order=(1, 0, 1, 24))
start = time.time()
model_fit = model.fit(dis=0)
print('fitting complete after {} seconds'.format(time.time()-start))
model_fit.summary()

f_steps = val_ts.shape[0]
results = model_fit.get_forecast(f_steps)

forecasts = pd.concat([results.predicted_mean, results.conf_int(alpha=0.05)], axis=1) 
forecasts.columns = ['Forecasts', 'Lower 95% CI', 'Upper 95% CI']

forecasts.head()
rmse = ((val_ts.values - results.predicted_mean)**2).mean()

plt.figure(figsize=(18,9))
plt.plot(train_ts[-24*7:], label='History (actual)')
plt.plot(val_ts, label='Future (actual)')
plt.plot(forecasts['Forecasts'], label='Forecast')
plt.fill_between(forecasts.index, forecasts['Lower 95% CI'], forecasts['Upper 95% CI'], color='0.8',label='95% confidence interval')
plt.legend()
plt.grid()
plt.title('RMSE: '+ str(rmse))
