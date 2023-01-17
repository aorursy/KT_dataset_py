import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6
df= pd.read_csv('../input/train_pCWxroh.csv')
Year= []

Month= []

Day= []

Hour=[]

def int_2_date(myDate):

    str_date=str(myDate)

    Year.append(int(str_date[0:4]))

    Month.append(int(str_date[4:6]))

    Day.append(int(str_date[6:8]))

    Hour.append(int(str_date[8:10]))




for i in range(0,len(df)):

    int_2_date(df['ID'][i])







df['Year']=pd.Series(Year, index=df.index)

df['Month']=pd.Series(Month, index=df.index)

df['Day']=pd.Series(Day, index=df.index)

df['Hour']=pd.Series(Hour, index=df.index)





from datetime import datetime

finalDate=[]

for i in range(0,len(df)):

    finalDate.append(datetime(df['Year'][i],df['Month'][i],df['Day'][i],df['Hour'][i],0))







df['Date']=pd.Series(finalDate, index=df.index)







del df['ID'],df['Year'],df['Month'],df['Day'],df['Hour']



df.set_index('Date', inplace=True)




df.head()



len(df)
split_point = len(df) - 2630

dataset, validation = df[0:split_point], df[split_point:]

dataset.dtypes
# prepare data

X = df.values

X = X.astype('float32')

train_size = int(len(X) * 0.50)

train, test = X[0:train_size], X[train_size:]
from sklearn.metrics import mean_squared_error

from math import sqrt

# walk-forward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

	# predict

	yhat = history[-1]

	predictions.append(yhat)

	# observation

	obs = test[i]

	history.append(obs)

	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# report performance

mse = mean_squared_error(test, predictions)

rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)
from pandas import Series

print(dataset.describe())
pyplot.figure(1)

pyplot.subplot(211)

dataset.hist()

pyplot.subplot(212)

dataset.plot(kind='kde')

pyplot.show()
from pandas import Series

from statsmodels.tsa.stattools import adfuller

from matplotlib import pyplot

 

# create a differenced series

def difference(dataset, interval=1):

	diff = list()

	for i in range(interval, len(dataset)):

		value = dataset[i] - dataset[i - interval]

		diff.append(value)

	return Series(diff)



X = dataset.values

X = X.astype('float32')

# difference data

months_in_year = 12

stationary = difference(X, months_in_year)

stationary.index = dataset.index[months_in_year:]

# check if stationary

result = adfuller(stationary)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))

# save

stationary.to_csv('stationary.csv')

# plot

stationary.plot()

pyplot.show()
from pandas import Series

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from matplotlib import pyplot

pyplot.figure()

pyplot.subplot(211)

plot_acf(stationary, ax=pyplot.gca())

pyplot.subplot(212)

plot_pacf(stationary, ax=pyplot.gca())

pyplot.show()
from pandas import DataFrame

from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot

 

# create a differenced series

def difference(dataset, interval=1):

	diff = list()

	for i in range(interval, len(dataset)):

		value = dataset[i] - dataset[i - interval]

		diff.append(value)

	return diff

 

# invert differenced value

def inverse_difference(history, yhat, interval=1):

	return yhat + history[-interval]

 

# load data

series = dataset

# prepare data

X = series.values

X = X.astype('float32')

train_size = int(len(X) * 0.50)

train, test = X[0:train_size], X[train_size:]

# walk-forward validation

history = [x for x in train]

predictions = list()

for i in range(len(test)):

	# difference data

	months_in_year = 12

	diff = difference(history, months_in_year)

	# predict

	model = ARIMA(diff, order=(0,0,1))

	model_fit = model.fit(trend='nc', disp=0)

	yhat = model_fit.forecast()[0]

	yhat = inverse_difference(history, yhat, months_in_year)

	predictions.append(yhat)

	# observation

	obs = test[i]

	history.append(obs)

# errors

residuals = [test[i]-predictions[i] for i in range(len(test))]

residuals = DataFrame(residuals)

print(residuals.describe())

# plot

pyplot.figure()

pyplot.subplot(211)

residuals.hist(ax=pyplot.gca())

pyplot.subplot(212)

residuals.plot(kind='kde', ax=pyplot.gca())

pyplot.show()
from pandas import Series

from pandas import DataFrame

from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot

from sklearn.metrics import mean_squared_error

from math import sqrt

 

# create a differenced series

def difference(dataset, interval=1):

	diff = list()

	for i in range(interval, len(dataset)):

		value = dataset[i] - dataset[i - interval]

		diff.append(value)

	return diff

 

# invert differenced value

def inverse_difference(history, yhat, interval=1):

	return yhat + history[-interval]

 

# load data

series = dataset

# prepare data

X = series.values

X = X.astype('float32')

train_size = int(len(X) * 0.50)

train, test = X[0:train_size], X[train_size:]

# walk-forward validation

history = [x for x in train]

predictions = list()

bias = 0.141482

for i in range(len(test)):

	# difference data

	months_in_year = 12

	diff = difference(history, months_in_year)

	# predict

	model = ARIMA(diff, order=(0,0,1))

	model_fit = model.fit(trend='nc', disp=0)

	yhat = model_fit.forecast()[0]

	yhat = bias + inverse_difference(history, yhat, months_in_year)

	predictions.append(yhat)

	# observation

	obs = test[i]

	history.append(obs)

# report performance

mse = mean_squared_error(test, predictions)

rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)

# errors

residuals = [test[i]-predictions[i] for i in range(len(test))]

residuals = DataFrame(residuals)

print(residuals.describe())

# plot

pyplot.figure()

pyplot.subplot(211)

residuals.hist(ax=pyplot.gca())

pyplot.subplot(212)

residuals.plot(kind='kde', ax=pyplot.gca())

pyplot.show()
from pandas import Series

from statsmodels.tsa.arima_model import ARIMA

from scipy.stats import boxcox

import numpy

 

# monkey patch around bug in ARIMA class

def __getnewargs__(self):

	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

 

ARIMA.__getnewargs__ = __getnewargs__

 

# create a differenced series

def difference(dataset, interval=1):

	diff = list()

	for i in range(interval, len(dataset)):

		value = dataset[i] - dataset[i - interval]

		diff.append(value)

	return diff

 

# load data

series = dataset

# prepare data

X = series.values

X = X.astype('float32')

# difference data

months_in_year = 12

diff = difference(X, months_in_year)

# fit model

model = ARIMA(diff, order=(0,0,1))

model_fit = model.fit(trend='nc', disp=0)

# bias constant, could be calculated from in-sample mean residual

bias = 0.141482

# save model

model_fit.save('model.pkl')

numpy.save('model_bias.npy', [bias])
from pandas import Series

from statsmodels.tsa.arima_model import ARIMAResults

import numpy

 

# invert differenced value

def inverse_difference(history, yhat, interval=1):

	return yhat + history[-interval]

 

series = dataset

months_in_year = 12

model_fit = ARIMAResults.load('model.pkl')

bias = numpy.load('model_bias.npy')

yhat = float(model_fit.forecast()[0])

yhat = bias + inverse_difference(series.values, yhat, months_in_year)

print('Predicted: %.3f' % yhat)
validation.head()
from pandas import Series

from matplotlib import pyplot

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARIMAResults

from sklearn.metrics import mean_squared_error

from math import sqrt

import numpy

 

# create a differenced series

def difference(dataset, interval=1):

	diff = list()

	for i in range(interval, len(dataset)):

		value = dataset[i] - dataset[i - interval]

		diff.append(value)

	return diff

 

# invert differenced value

def inverse_difference(history, yhat, interval=1):

	return yhat + history[-interval]

 

# load and prepare datasets

X = dataset.values.astype('float32')

history = [x for x in X]

months_in_year = 12

y = validation.values.astype('float32')

# load model

model_fit = ARIMAResults.load('model.pkl')

bias = numpy.load('model_bias.npy')

# make first prediction

predictions = list()

yhat = float(model_fit.forecast()[0])

yhat = bias + inverse_difference(history, yhat, months_in_year)

predictions.append(yhat)

history.append(y[0])

print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

# rolling forecasts

for i in range(1, len(y)):

	# difference data

	months_in_year = 12

	diff = difference(history, months_in_year)

	# predict

	model = ARIMA(diff, order=(0,0,1))

	model_fit = model.fit(trend='nc', disp=0)

	yhat = model_fit.forecast()[0]

	yhat = bias + inverse_difference(history, yhat, months_in_year)

	predictions.append(yhat)

	# observation

	obs = y[i]

	history.append(obs)

	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# report performance

mse = mean_squared_error(y, predictions)

rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)

pyplot.plot(y)

pyplot.plot(predictions, color='red')

pyplot.show()
mytestData=pd.read_csv('../input/test_bKeE5T8.csv')
Year= []

Month= []

Day= []

Hour=[]

def int_2_date(myDate):

    str_date=str(myDate)

    Year.append(int(str_date[0:4]))

    Month.append(int(str_date[4:6]))

    Day.append(int(str_date[6:8]))

    Hour.append(int(str_date[8:10]))
for i in range(0,len(mytestData)):

    int_2_date(mytestData['ID'][i])
mytestData['Year']=pd.Series(Year, index=mytestData.index)

mytestData['Month']=pd.Series(Month, index=mytestData.index)

mytestData['Day']=pd.Series(Day, index=mytestData.index)

mytestData['Hour']=pd.Series(Hour, index=mytestData.index)
from datetime import datetime

finalDate=[]

for i in range(0,len(mytestData)):

    finalDate.append(datetime(mytestData['Year'][i],mytestData['Month'][i],mytestData['Day'][i],mytestData['Hour'][i],0))
mytestData['Date']=pd.Series(finalDate, index=mytestData.index)
del mytestData['ID'],mytestData['Year'],mytestData['Month'],mytestData['Day'],mytestData['Hour']
mytestData.set_index('Date', inplace=True)
mytestData.head()
mytestData=mytestData['Count'].fillna('0')
mytestData.head()
from pandas import Series

from matplotlib import pyplot

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARIMAResults

from sklearn.metrics import mean_squared_error

from math import sqrt

import numpy

 

# create a differenced series

def difference(dataset, interval=1):

	diff = list()

	for i in range(interval, len(dataset)):

		value = dataset[i] - dataset[i - interval]

		diff.append(value)

	return diff

 

# invert differenced value

def inverse_difference(history, yhat, interval=1):

	return yhat + history[-interval]

 

# load and prepare datasets

X = df.values.astype('float32')

history = [x for x in X]

months_in_year = 12

y = mytestData.values.astype('float32')

# load model

model_fit = ARIMAResults.load('model.pkl')

bias = numpy.load('model_bias.npy')

# make first prediction

predictions = list()

yhat = float(model_fit.forecast()[0])

yhat = bias + inverse_difference(history, yhat, months_in_year)

predictions.append(yhat)

history.append(yhat)

print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

# rolling forecasts

for i in range(1, len(y)):

	# difference data

	months_in_year = 12

	diff = difference(history, months_in_year)

	# predict

	model = ARIMA(diff, order=(0,0,1))

	model_fit = model.fit(trend='nc', disp=0)

	yhat = model_fit.forecast()[0]

	yhat = bias + inverse_difference(history, yhat, months_in_year)

	predictions.append(yhat)

	# observation

	obs = yhat

	history.append(obs)

	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# report performance

mse = mean_squared_error(y, predictions)

rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)

pyplot.plot(y)

pyplot.plot(predictions, color='red')

pyplot.show()
len(predictions)
len(mytestData)
mytestData.head()
submission=pd.read_csv('../input/sample_submission_EjnOGo9.csv')
submission['Count']=predictions
submission.head()
submission.to_csv('myresult.csv')
from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))
print(check_output(["ls", "../input"]).decode("utf8"))
submission.head()
int(submission['Count'])
submission['Count'].astype(int)
submission.to_csv('../working/pakka.csv')
from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))