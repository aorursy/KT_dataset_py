!pip install --user statsmodels
%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import datetime



%config InlineBackend.figure_format = 'retina'
df = pd.read_csv('../input/week4data/1NewAAPL10Y.csv')



df['date'] = pd.to_datetime(df['date'])

df.sort_values('date', inplace=True)

df.set_index('date', inplace=True)



print(df.shape)



df.head()
df_week = df.resample('w').mean()

df_week = df_week[['close']]

df_week.head()
df_week['weekly_ret'] = np.log(df_week['close']).diff()

df_week.head()
# drop null rows

df_week.dropna(inplace=True)
df_week.weekly_ret.plot(kind='line', figsize=(12, 6));
udiff = df_week.drop(['close'], axis=1)

udiff.head()
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
rolmean = udiff.rolling(20).mean()

rolstd = udiff.rolling(20).std()
plt.figure(figsize=(12, 6))

orig = plt.plot(udiff, color='blue', label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std Deviation')

plt.title('Rolling Mean & Standard Deviation')

plt.legend(loc='best')

plt.show(block=False)
# Perform Dickey-Fuller test

dftest = sm.tsa.adfuller(udiff.weekly_ret, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

for key, value in dftest[4].items():

    dfoutput['Critical Value ({0})'.format(key)] = value

    

dfoutput
from statsmodels.graphics.tsaplots import plot_acf



# the autocorrelation chart provides just the correlation at increasing lags

fig, ax = plt.subplots(figsize=(12,5))

plot_acf(udiff.values, lags=10, ax=ax)

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf



fig, ax = plt.subplots(figsize=(12,5))

plot_pacf(udiff.values, lags=10, ax=ax)

plt.show()
from statsmodels.tsa.arima_model import ARMA



# Notice that you have to use udiff - the differenced data rather than the original data. 

ar1 = ARMA(tuple(udiff.values), (3, 1)).fit()

ar1.summary()
plt.figure(figsize=(12, 8))

plt.plot(udiff.values, color='blue')

preds = ar1.fittedvalues

plt.plot(preds, color='red')

plt.show()
steps = 2



forecast = ar1.forecast(steps=steps)[0]



plt.figure(figsize=(12, 8))

plt.plot(udiff.values, color='blue')



preds = ar1.fittedvalues

plt.plot(preds, color='red')



plt.plot(pd.DataFrame(np.array([preds[-1],forecast[0]]).T,index=range(len(udiff.values)+1, len(udiff.values)+3)), color='green')

plt.plot(pd.DataFrame(forecast,index=range(len(udiff.values)+1, len(udiff.values)+1+steps)), color='green')

plt.title('Display the predictions with the ARIMA model')

plt.show()