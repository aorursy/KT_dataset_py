import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib

import matplotlib.pyplot as plt

from datetime import datetime

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
df = pd.read_csv("../input/AirPassengers.csv")

df.head()
df['Month'].min(), df['Month'].max()
df.dtypes
df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
df = df.set_index('Month')

df.index
training_set = df[:'1959']

training_set.index

test_set = df['1960':]

test_set.index
training_set.plot(figsize=(15, 6))

plt.show()
test_set.plot(figsize=(15, 6))

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(training_set, model='additive')

fig = decomposition.plot()

plt.show()
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(training_set,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
mod = sm.tsa.statespace.SARIMAX(training_set,

                                order=(1, 0, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('1959-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = training_set['1949':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()
y_forecasted = pd.DataFrame(data=pred.predicted_mean, columns =['y_forecasted'])

y_forecasted.head()

y_truth = training_set['1959-01':]

y_truth.dtypes

mse = ((y_forecasted.values -  - y_truth.values) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=12)

pred_ci = pred_uc.conf_int()



ax = training_set.plot(label='observed', figsize=(14, 7))



testSet = test_set.plot(label='Test Set', figsize=(14, 7))



pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

pred_uc.predicted_mean.plot(ax=testSet, label='Forecast')



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()





