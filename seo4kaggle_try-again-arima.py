# https://machinelearningmastery.com/make-sample-forecasts-arima-python/
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
# line plot of time series

from pandas import Series

from matplotlib import pyplot

# load dataset

series = Series.from_csv('../input/daily-minimum-temperatures.csv', header=0)

# display first few rows

print(series.head(20))

# line plot of dataset

series.plot()

pyplot.show()
# split the dataset

split_point = len(series) - 7

dataset, validation = series[0:split_point], series[split_point:]

print('Dataset %d, Validation %d' % (len(dataset), len(validation)))

# create a differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return np.array(diff)
# invert differenced value

def inverse_difference(history, yhat, interval=1):

    return yhat + history[-interval]
differenced[3260:3280]
from statsmodels.tsa.arima_model import ARIMA



# load dataset

series = dataset

# seasonal difference

X = series.values

days_in_year = 365

differenced = difference(X, days_in_year)

# fit model

model = ARIMA(differenced, order=(12,1,0))

model_fit = model.fit(disp=0)

# print summary of fit model

print(model_fit.summary())

# one-step out-of sample forecast

forecast = model_fit.forecast(steps=7)[0]
X[len(X) -365]
# invert the differenced forecast to something usable

forecast = inverse_difference(X, forecast, days_in_year)

print('Forecast: %f' % forecast)
# one-step out of sample forecast

start_index = len(differenced)

end_index = len(differenced)

forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable

forecast = inverse_difference(X, forecast, days_in_year)

print('Forecast: %f' % forecast)
model_fit.forecast(steps=7)
# multi-step out-of-sample forecast

forecast = model_fit.forecast(steps=7)[0]

# invert the differenced forecast to something usable

history = [x for x in X]

day = 1

for yhat in forecast:

	inverted = inverse_difference(history, yhat, days_in_year)

	print('Day %d: %f' % (day, inverted))

	history.append(inverted)

	day += 1
start_index = len(differenced)

end_index = start_index + 6

forecast = model_fit.predict(start=start_index, end=end_index)

# invert the differenced forecast to something usable

history = [x for x in X]

day = 1

for yhat in forecast:

	inverted = inverse_difference(history, yhat, days_in_year)

	print('Day %d: %f' % (day, inverted))

	history.append(inverted)

	day += 1