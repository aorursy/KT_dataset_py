# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from statsmodels.tsa.stattools import acf,pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
from pandas import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#from sklearn.metrics import mean_squared_error
#pip install pmdarima
#from pmdarima.arima.utils import ndiffs
#import pmdarima as pm
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6
df=pd.read_csv("/kaggle/input/predice-el-futuro/train_csv.csv")
df.head()
df.tail()
df.describe()
plt.plot(df['feature'])
import statsmodels.graphics.tsaplots

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.feature); axes[0, 0].set_title('Original Series')
statsmodels.graphics.tsaplots.plot_acf(df.feature, ax=axes[0, 1])


# 1st Differencing
axes[1, 0].plot(df.feature.diff()); axes[1, 0].set_title('1st Order Differencing')
statsmodels.graphics.tsaplots.plot_acf(df.feature.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.feature.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
statsmodels.graphics.tsaplots.plot_acf(df.feature.diff().diff().dropna(), ax=axes[2, 1])

plt.show()
result = ts.adfuller(df.feature.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
#PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.feature.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
statsmodels.graphics.tsaplots.plot_pacf(df.feature.diff().dropna(), ax=axes[1])

plt.show()
# 1,1,1 ARIMA Model
#model = ARIMA(df.feature, order=(1,1,1))
# 1,0,0 ARIMA Model
model = ARIMA(df.feature, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Actual vs Fitted
model_fit.plot_predict(dynamic=False,plot_insample=True)
plt.show()
# Create Training and Test
train = df.feature[0:56]
test = df.feature[56:]
train, test
train.plot( title= 'All day features', fontsize=7) 
test.plot( title= 'All day features', fontsize=7) 
plt.show()
# Build Model 
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit(disp=0)  

# Forecast
fc, se, conf = fitted.forecast(24, alpha=0.05)  

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({ 'rmse':rmse})

forecast_accuracy(fc, test)
# Build Model 
model = ARIMA(train, order=(2,1,4))
fitted = model.fit(disp=0)  

# Forecast
fc, se, conf = fitted.forecast(24, alpha=0.05)  

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
#Accuracy metrics
def forecast_accuracy(forecast, actual):
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({ 'rmse':rmse})

forecast_accuracy(fc, test)

