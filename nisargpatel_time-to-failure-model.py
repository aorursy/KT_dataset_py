!pip install pmdarima

!pip install arch
import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib import pyplot

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

import datetime as datetime

import time as time

from tqdm import tqdm

from collections import defaultdict



## Time Series libraries

from scipy import stats

from statsmodels import tsa

from statsmodels.graphics import tsaplots

from statsmodels.tsa.stattools import acf

from statsmodels.tsa.stattools import adfuller

from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.tsa.seasonal import seasonal_decompose



## Auto Arima library

import pmdarima as pm

from pmdarima.arima import ARIMA



## Arma library

from statsmodels.tsa.arima_model import ARMA 

from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model



## linear regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error



## ignore warnings

import warnings

warnings.filterwarnings('ignore')



## math libraries

from numpy import log

from sklearn.metrics import mean_squared_error
data = pd.read_csv('../input/timeseriesdata/x_train_105.csv',index_col=[0])

target = pd.read_csv('../input/timeseriesdata/y_train_105.csv',index_col=[0])

print('X:',data.shape)

print('Y:',target.shape)
data['seismic_average'] = data['average']

data = pd.DataFrame(data['seismic_average'])

#data.dropna(inplace=True)

data.head()
len_data = data.shape[0]

train_split = 0.95



split_size=int(len_data*train_split)

avg_xtrain, avg_xtest = data[:split_size],data[split_size:]

avg_ytrain,avg_ytest= target[:split_size],target[split_size:]
print("X_train : ",avg_xtrain.shape)

print("y_train : ",avg_ytrain.shape)

print("X_test : ",avg_xtest.shape)

print("y_test : ",avg_ytest.shape)
plt.plot(avg_ytrain)

plt.title("Time Plot - Time_to_failure")
#results_dict

#max_value = []

#for i in tqdm(range(len(avg_ytrain))):

#    max_value = avg_ytrain[i:i+1]/16.094698

#    results_dict['min_value'].append(max_value)
#results_dict = pd.DataFrame(results_dict)

#new = results_dict['min_value'].str.split(" ")

#results_dict["minvalue_1"]= new[0] 

#esults_dict["minvalue_2"]= new[1] 

#results_dict["minvalue_3"]= new[2] 
#results_dict
plt.plot(avg_ytrain.diff())

plt.xticks(np.arange(0,10000, step= 500))

plt.title("Time Plot - Time_to_failure Difference")
nlags = 1600

fig = plt.figure(figsize=(12,8))

ax1 = plt.subplot(2, 2, 3) 

fig = tsaplots.plot_acf(avg_ytrain.diff().dropna(), lags=nlags, ax=ax1,title='ACF for Time_to_failure Difference')

plt.xticks(np.arange(0,1600, step= 400))



ax2 = plt.subplot(2, 2, 4)

fig = tsaplots.plot_pacf(avg_ytrain.diff().dropna(), lags=nlags, ax=ax2,title='PACF for Time_to_failure Difference')

plt.xticks(np.arange(0,1600, step= 400))



ax3 = plt.subplot(2, 2, 1)

fig = tsaplots.plot_acf(avg_ytrain, lags=nlags, ax=ax3,title='ACF for Time_to_failure')

plt.xticks(np.arange(0,1600, step= 400))



ax4 = plt.subplot(2, 2, 2)

fig = tsaplots.plot_pacf(avg_ytrain, lags=nlags, ax=ax4,title='PACF for Time_to_failure')

plt.xticks(np.arange(0,1600, step= 400))

plt.show()
# Ljung-box test for auto-correlation

_,p=acorr_ljungbox(avg_ytrain.diff().dropna(),lags=10)

print('Ljung-Box test p-values for 10-lags: ',p)
data1 = avg_ytrain.diff().dropna()

data1 = data1.iloc[:,0].values

result = adfuller(data1)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))
model_auto = pm.auto_arima(avg_ytrain, error_action='ignore', trace= True, seasonal=True, 

                            information_criterion='bic')
model_auto.summary()
model = SARIMAX(avg_ytrain, order=(0,1,0))

model_fit = model.fit()
# Ljung-box test for auto-correlation

_,p=acorr_ljungbox(model_fit.resid,lags=10)

print('Ljung-Box test p-values for 10-lags: ',p)
plt.plot(model_fit.resid)
plt.plot(model_fit.forecast(steps= len(avg_ytest)))

plt.plot(avg_ytest)
pred = model_fit.forecast(steps= len(avg_ytest))
mean_absolute_error(avg_ytest,pred)