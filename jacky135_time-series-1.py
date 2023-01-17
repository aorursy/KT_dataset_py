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
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train['Datetime']

test.index  = test['Datetime']
train.drop(['ID','Datetime'],axis  = 1,inplace = True)

test.drop(['Datetime'],axis = 1,inplace = True)
train
Train=train.ix['2012-08-25':'2014-06-24'] 

valid=train.ix['2014-06-25':'2014-09-25']
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_avg = valid.copy()

fit1 = ExponentialSmoothing(Train['Count'],seasonal_periods=576,trend='add', seasonal='add').fit(use_boxcox=True) 

y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter') 

plt.legend(loc='best') 

plt.show()

from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter)) 

print(rms)
submission = test.copy()

submission['Count'] = fit1.predict(start="2014-9-26", end="2015-4-27")

submission.to_csv("Time Series.csv",index=False)
y_hat_avg_holt = valid.copy()

fit2 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.267,smoothing_slope = 0.1325)

y_hat_avg_holt['Holt_Winter'] = fit2.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg_holt['Holt_Winter'], label='Holt_Winter') 

plt.legend(loc='best') 

plt.show()

from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(valid.Count, y_hat_avg_holt.Holt_Winter)) 

print(rms)