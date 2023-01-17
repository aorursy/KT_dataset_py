import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
GOOG = pd.read_csv("../input/GOOGL_2006-01-01_to_2018-01-01.csv",index_col = "Date",parse_dates = ['Date'])

GOOG.describe()
GOOG.sample(5)
import matplotlib.pyplot as plt

import numpy as np

from pylab import rcParams
plt.figure(figsize=(20,6))

plt.plot(GOOG['Close'])
goog_close_log = np.log10(GOOG['Close'])

plt.figure(figsize=(20,6))

plt.plot(goog_close_log)
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams['figure.figsize'] = 11, 9

decomposition = seasonal_decompose(GOOG['Close'], model='additive',freq=360)

plt.figure(figsize=(20,6))

fig = decomposition.plot()

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams['figure.figsize'] = 11, 9

decomposition = seasonal_decompose(GOOG['Close'], model='multiplicative',freq=360)

plt.figure(figsize=(20,6))

fig = decomposition.plot()

plt.show()
#An introduction to moving average

a = [1,2,1]

a = pd.DataFrame({'a':a})

a.rolling(window = 2).mean()
moving_average100 = GOOG['Close'].rolling(window=100).mean()

plt.figure(figsize=(20,8))

plt.plot(GOOG['Close'], label='Google', color='orange')

plt.plot(moving_average100, label='Google MA 100', color='blue')

plt.legend(loc='upper left')

plt.show()
exp_ma50 = GOOG['2010':]['Close'].ewm(span=50, adjust=False).mean()

exp_ma100 = GOOG['2010':]['Close'].ewm(span=200, adjust=False).mean()





plt.figure(figsize=(20,8))

plt.plot(exp_ma50, label='GOOG 50 day EMA')

plt.plot(exp_ma100, label='GOOG 100 day EMA')

plt.legend(loc='upper left')

plt.show()



#For more information about ewm, visit https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html
plt.figure(figsize=(20,6))

x = GOOG['Close'][0:len(GOOG['Close'])-1].values

y = GOOG['Close'][1:].values

x_index = np.arange(100,1200,50)

plt.xticks(x_index)

plt.scatter(x,y)

plt.show()
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error

import math as m



X = GOOG['Close'].values



#Assigning 80% of data points to test and 20% for test

train, test = X[0:m.floor(len(X)*0.8)], X[m.floor(len(X)*0.8):]



# train autoregression

model = AR(train)

model_fit = model.fit()



print('Lag: ',model_fit.k_ar)

print('Coefficients: ',model_fit.params)



# make predictions

predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

error = mean_squared_error(test, predictions)



#print('Test MSE: %.3f' % error)

print('Mean Square Error: ', error)



# plot results

plt.plot(test)

plt.plot(predictions, color='red')

plt.show()
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(GOOG['Close'], lags=20)

plt.show()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize = (20,6))

autocorrelation_plot(GOOG['Close'])

plt.show()
from statsmodels.tsa.stattools import adfuller

result = adfuller(GOOG['Close'])

result_ = pd.Series(result[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

for key, value in result[4].items():

    result_['Critical Value (%s)' %key] = value

print(result_)
from statsmodels.tsa.arima_model import ARIMA

rcParams['figure.figsize'] = 16, 6

#model = ARIMA(GOOG["Close"].diff().iloc[1:].values, order=(2,1,0))

model = ARIMA(GOOG["Close"], order=(2,1,1))

result = model.fit()

print(result.summary())

result.plot_predict(start=700, end=1000)

plt.show()