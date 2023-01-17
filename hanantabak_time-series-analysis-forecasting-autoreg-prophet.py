# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas import Grouper

from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.ar_model import AutoReg

from fbprophet import Prophet

from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics

from fbprophet.plot import plot_cross_validation_metric









# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', header=0, parse_dates=[0],index_col=0, squeeze=True)
data.head()
data.size
type(data)
print(data['1987-01'])
data.describe()
plt.figure(figsize=(20,10))

data.plot(style='k-')
plt.figure(figsize=(20,10))

data.plot(style='k.')
groups = data.groupby(Grouper(freq='A'))

years = pd.DataFrame()

for name,group in groups:

    years[name.year] = group.values

plt.figure(figsize=(10,6))

years.plot(subplots=True, legend=False)

plt.show()
data.hist()
data.plot(kind='kde')
years.boxplot()
last_year = data['1990']

groups = last_year.groupby(Grouper(freq='M'))

months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)

months.columns = range(1,13)

months.boxplot()

months

years
plt.figure(figsize=(10,6))

plt.matshow(years.T,interpolation=None, aspect='auto')
plt.figure(figsize=(10,6))

plt.matshow(months,interpolation=None, aspect='auto')
lag_plot(data)
lag_plot(months)
autocorrelation_plot(data)
values = pd.DataFrame(data.values)

df = pd.concat([values.shift(1),values],axis=1)

df.columns = ['t-1','t+1']

df.head()
X = df.values

train_size = int(len(X)*0.7)

train, test = X[1:train_size], X[train_size:]

train_X,train_y = train[:,0],train[:,1]

test_X,test_y = test[:,0],test[:,1]

def model_persistence(x):

    return x

pred = list()

for x in test_X:

    y1= model_persistence(x)

    pred.append(y1)

    

error = mean_squared_error(test_y,pred)

print('MSE : %.3f' % error)

plt.figure(figsize=(20,10))

plt.plot(train_y)

plt.plot([None for i in train_y]+[x for x in test_y])

plt.plot([None for i in train_y]+[x for x in pred])



df.corr()
plot_acf(data, lags=40)
X = data.values

train,test = X[1:len(X)-7],X[len(X)-7:]

model = AutoReg(train, lags=29)

model_fit = model.fit()

coef = model_fit.params

print(coef)
pred = model_fit.predict(start= len(train), end=len(train)+len(test)-1, dynamic=False)

for x in range(len(pred)):

    print('Predicted temperature is %f , while actual temp is %f'%(pred[x],test[x]))
mse = mean_squared_error(pred,test)

print('MSE is %f' %mse)
plt.plot(test)

plt.plot(pred, color='red')
window = 29

model = AutoReg(train, lags=29)

model_fit = model.fit()

coef = model_fit.params

# walk forward over time steps in test

history = train[len(train)-window:]

history = [history[i] for i in range(len(history))]

predictions = list()

for t in range(len(test)):

	length = len(history)

	lag = [history[i] for i in range(length-window,length)]

	yhat = coef[0]

	for d in range(window):

		yhat += coef[d+1] * lag[window-d-1]

	obs = test[t]

	predictions.append(yhat)

	history.append(obs)

	print('predicted=%f, expected=%f' % (yhat, obs))

rmse = mean_squared_error(test, predictions)

print('Test RMSE: %.3f' % rmse)

# plot

plt.plot(test)

plt.plot(predictions, color='red')

data_df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')

data_df = data_df.rename(columns = {'Date':'ds','Temp':'y'}, inplace=False)

data_df.head()
model = Prophet()

model.fit(data_df)

future = model.make_future_dataframe(periods=365)

future.shape, data_df.shape
future.head(),data_df.head(),future.tail(),data_df.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
model.plot(forecast)
model.plot_components(forecast)
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')

df_cv.head()
df_p = performance_metrics(df_cv)

df_p.head()
fig = plot_cross_validation_metric(df_cv, metric='rmse')