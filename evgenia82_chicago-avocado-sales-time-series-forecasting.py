# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import itertools

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/avocado.csv')

df.head()
df.info()
df = df.drop(['Unnamed: 0'], axis = 1)
df["Date"] = pd.to_datetime(df["Date"])
print(df['region'].nunique())

df['region'].value_counts().tail()
Chicago_df = df[df['region']=='Chicago']

Chicago_df.head()
Chicago_df.set_index('Date', inplace = True)
Chicago_df = Chicago_df.sort_values(by = 'Date')

Chicago_df.head()
Chicago_df['AveragePrice'].plot(figsize = (18,6))
Chicago_df['AveragePrice'].rolling(25).mean().plot(figsize = (18,6))
print(Chicago_df['AveragePrice'].max())

Chicago_df[Chicago_df['AveragePrice']==2.3]
print(Chicago_df['AveragePrice'].min())

Chicago_df[Chicago_df['AveragePrice']==0.7]
#Chicago_df[Chicago_df.index=='2017-02-05']
print(Chicago_df.index.min())

print(Chicago_df.index.max())
columns = ['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','type',

           'year','region']

Chicago_df.drop(columns, axis =1, inplace=True)
Chicago_df.isnull().sum()
Chicago_df = Chicago_df.groupby('Date')['AveragePrice'].sum().reset_index()
Chicago_df = Chicago_df.set_index('Date')

Chicago_df.index
y = Chicago_df['AveragePrice']
from pylab import rcParams

rcParams['figure.figsize'] = 18,8

decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')

fig = decomposition.plot()

plt.show()
p = d = q = range(0,2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d,q))]
print('Examples of parameter control combinations for Seasonal ARIMA...')

print('SARIMAX:{} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX:{} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX:{} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX:{} x {}'.format(pdq[2], seasonal_pdq[4]))
# fix the error

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal,

                                            enforce_stationarity=False, enforce_invertibility=False)

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
results.plot_diagnostics(figsize =(16,8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2015':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color = 'k', alpha = 0.2)

ax.set_xlabel('Date')

ax.set_ylabel('Avocado Sales')

plt.legend()

plt.show()
y_forecasted = pred.predicted_mean

y_truth = y['2017-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Avocado Sales')

plt.legend()

plt.show()