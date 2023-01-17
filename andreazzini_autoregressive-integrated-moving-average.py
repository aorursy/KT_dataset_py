# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/realAWSCloudwatch/realAWSCloudwatch"]).decode("utf8"))
fpath = "../input/realAWSCloudwatch/realAWSCloudwatch/"

fname = "ec2_cpu_utilization_825cc2.csv"



fullPath = fpath + fname



data = pd.read_csv(fullPath)

data.head()
x = [dt.datetime.strptime(d,"%Y-%m-%d %H:%M:%S").date() for d in data["timestamp"]]

y = data["value"]



plt.plot(x,y)

plt.show()
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot

from pandas import DataFrame

 

def parser(x):

	return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

 

data = pd.read_csv(fullPath, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)



arimaM = ARIMA(data, order=(5,1,0))

arimaMfit = arimaM.fit(disp=0)

print(arimaMfit.summary())

# plot residual errors

errors = DataFrame(arimaMfit.resid)

errors.plot()

pyplot.show()

errors.plot(kind='kde')

pyplot.show()

print(errors.describe())
from pandas import datetime

from matplotlib import pyplot

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

 

def parser(x):

	return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

 

data = pd.read_csv(fullPath, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

X = data.values

size = int(len(X) * 0.66)

limitCount = 50

train, test = X[0:size], X[size:size+limitCount]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

	model = ARIMA(history, order=(5,1,0))

	model_fit = model.fit(disp=0)

	output = model_fit.forecast()

	yhat = output[0]

	predictions.append(yhat)

	obs = test[t]

	history.append(obs)

	print('pred=%f, exp=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Mean Squared Error: %.3f' % error)

# plot

pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()