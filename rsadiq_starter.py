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
data = pd.read_csv('/kaggle/input/ml-starter/sale_data_train.csv')

data.head()
import matplotlib.pyplot as plt

data.plot()

plt.show()
data.InvoiceDate = pd.to_datetime(data.InvoiceDate, format = '%Y-%M-%d').dt.date

data.InvoiceDate.dtype
from statsmodels.tsa.stattools import adfuller

X = data.TotalSales.values

result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))
train = data[:180]

test = data[181:]

print(train.shape)

print(test.shape)
y_hat_avg = test.copy()

y_hat_avg['moving_avg_forecast'] = train['TotalSales'].rolling(20).mean().iloc[-1]

plt.figure(figsize=(16,8))

plt.plot(train['TotalSales'], label='Train')

plt.plot(test['TotalSales'], label='Test')

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')

plt.legend(loc='best')

plt.show()
from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(test.TotalSales, y_hat_avg.moving_avg_forecast))

print('RMSE Moving Avg:',rms)
sub = pd.read_csv('/kaggle/input/ml-starter/sample_submission.csv')

print(sub.shape)
y_hat_avg = sub.copy()

y_hat_avg['moving_avg_forecast'] = data['TotalSales'].rolling(5).mean().iloc[-1]

sub['TotalSales'] = y_hat_avg['moving_avg_forecast']

sub.to_csv('submission.csv',index = False)

sub.head()
from statsmodels.tsa.api import SimpleExpSmoothing

y_ses = test.copy()

ses_model = SimpleExpSmoothing(np.asarray(train['TotalSales'])).fit(smoothing_level=0.6, optimized=False)

y_ses['SES'] = ses_model.forecast(len(test))



rms = sqrt(mean_squared_error(test.TotalSales, y_ses.SES))

print('RMSE Moving Avg:',rms)



plt.figure(figsize=(16,8))

plt.plot(train['TotalSales'], label='Train')

plt.plot(test['TotalSales'], label='Test')

plt.plot(y_ses['SES'], label='SES')

plt.legend(loc='best')

plt.show()
sub['TotalSales'] = ses_model.forecast(len(sub))

sub.to_csv('submission_ses.csv',index = False)

sub.head()
from statsmodels.tsa.arima_model import ARIMA

model_arima = ARIMA(train.TotalSales, order=(0, 0, 0)).fit(disp=0)

print(model_arima.summary())
y_arima = test.copy()

y_arima['ARIMA'] = model_arima.forecast(len(test))[0]



rms = sqrt(mean_squared_error(test.TotalSales, y_arima.ARIMA))

print('RMSE Moving Avg for ARIMA:',rms)



plt.figure(figsize=(16,8))

plt.plot( train['TotalSales'], label='Train')

plt.plot(test['TotalSales'], label='Test')

plt.plot(y_arima['ARIMA'], label='ARIMA')

plt.legend(loc='best')

plt.show()
sub['TotalSales'] = model_arima.forecast(len(sub))[0]

sub.to_csv('submission_arima.csv',index = False)

sub.head()