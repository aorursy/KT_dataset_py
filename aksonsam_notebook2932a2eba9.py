# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas import Series

series = Series.from_csv('../input/Mat1.csv', header=0)
series.head()
series['1948']
split_ = len(series) - 12
dataset, validation = series[0:split_], series[split_:]
len(dataset)


len(validation)
from matplotlib import pyplot

series.plot()

pyplot.show()
#preparing train and test data

data = series.values

data = data.astype('float32')

train_size = int(len(data)* 0.50)

train, test = data[0:train_size], data[train_size:]
from statsmodels.tsa.stattools import adfuller



def differenced(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i-interval]

        diff.append(value)

    return Series(diff)
months =12

stationary = differenced(data, months)

stationary.index = series.index[months:]
#check stationary

result = adfuller(stationary)

print('ADF Statistic: %f' % result[0])

print('p-value : %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' %(key, value))
def inverse_difference(history, y, interval=1):

    return y+ history[-interval]
#ACF, PACF

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



pyplot.figure()

pyplot.subplot(211)

plot_acf(series, ax=pyplot.gca())

pyplot.subplot(212)

plot_pacf(series, ax=pyplot.gca())

pyplot.show()
from statsmodels.tsa.arima_model import ARIMA

from pandas import DataFrame



history = [x for x in train]

predictions = list()

for i in range(len(test)):

    months = 12

    diff = differenced(history, months)

    #predict

    model = ARIMA(diff, order=(0,0,1))

    model_fit = model.fit(trend='nc', disp=0)

    y = model_fit.forecast()[0]

    y = inverse_difference(history, y, month)

    predictions.append(y)

    #observations

    obs = test[i]

    history.append(obs)

residuals = [test[i]-predictions[i] for i in range(len(test))]

residuals = DataFrame(residuals)

print(residuals.describe())



#plot

pyplot.figure()

pyplot.subplot(211)

residuals.hist(ax=pyplot.gca())

pyplot.subplot(212)

residuals.plot(kind='kde', ax=pyplot.gca())

pyplot.show()