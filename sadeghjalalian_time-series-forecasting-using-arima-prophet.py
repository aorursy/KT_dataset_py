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
import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import matplotlib

plt.style.use('ggplot')

import warnings

import itertools

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
df = pd.read_csv('/kaggle/input/accident_UK.csv')

df.head()
df.info()
df['Date'] = pd.to_datetime(df['Date'])

df.head()
df = df.sort_values(by=['Date'])

df.head()
df.info()
accident = df.set_index('Date')

accident.index
y = accident['Total_Accident'].resample('MS').mean()

y.head()
y.plot(figsize=(15, 6))

plt.show()
from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 16, 10

decomposition = sm.tsa.seasonal_decompose(y, model='additive')

fig = decomposition.plot()

plt.show()
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)



            results = mod.fit()

            

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()
y_forecasted = pred.predicted_mean

y_truth = y['2017-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()
df = df.sort_values(by=['Date'])

df.head()
df = df.rename(columns={'Date': 'ds',

                        'Total_Accident': 'y'})

df.head()
ax = df.set_index('ds').plot(figsize=(15, 8))

ax.set_ylabel('Total Accident')

ax.set_xlabel('Date')



plt.show()
from fbprophet import Prophet

my_model = Prophet(interval_width=0.95)

my_model.fit(df)
future_dates = my_model.make_future_dataframe(periods=36, freq='MS')

future_dates.tail()
forecast = my_model.predict(future_dates)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
pd.plotting.register_matplotlib_converters()

my_model.plot(forecast, uncertainty=True)
my_model.plot_components(forecast)
from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(my_model, initial='730 days', period='180 days', horizon = '365 days')

df_cv.head()
from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(df_cv)

df_p.head()