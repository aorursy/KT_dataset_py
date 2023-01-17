import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
temp = pd.read_csv('../input/temperature.csv', parse_dates=['datetime'])

temp = temp.set_index('datetime')

print('Dataset shape: {}'.format(temp.shape))

temp.head()
all_std = temp.std(axis=0)

max_std = all_std.max()

city_max_std = temp.columns[all_std==max_std][0]



print('City with highest temperature variation: {} ({} degrees)'.format(city_max_std,round(max_std,2)))
data = temp[['San Francisco','Minneapolis']]

data.describe()
data = data-273.15

data.describe()
_=data.plot(

    figsize=(15,5),

    subplots=False,

    title='Temperature',

    alpha=0.7

)

_=plt.xlabel('Date')

_=plt.ylabel('Temperature')
SF_non_missing = data['San Francisco'].dropna()

max_date = SF_non_missing.index.max()

data = data[data.index <= max_date]
print(data.isna().sum())
data_mean = data.resample('D').mean()

data_min = data.resample('D').min()

data_max = data.resample('D').max()

print('Resample shape: {}'.format(data_mean.shape))

data_mean.describe()
print('Missing data now?')

print(data_mean.isna().sum())
_=data_mean.plot(

    figsize=(15,5),

    subplots=False,

    title='Temperature',

    alpha=0.7

)

_=plt.fill_between(

    x=data_mean.index,

    y1=data_min['San Francisco'].values,

    y2=data_max['San Francisco'].values,

    alpha=0.3

)

_=plt.fill_between(

    x=data_mean.index,

    y1=data_min['Minneapolis'].values,

    y2=data_max['Minneapolis'].values,

    color='orange',

    alpha=0.3

)

_=plt.xlabel('Date')

_=plt.ylabel('Temperature')
_=plt.hist(data_mean['San Francisco'], alpha=0.5, label='San Francisco')

_=plt.hist(data_mean['Minneapolis'], alpha=0.5, label='Minneapolis')

_=plt.legend()
cut = data_mean.index[int(0.5*len(data_mean))]

print('Mean before {}:'.format(cut))

print(data_mean.loc[:cut].mean())

print('')

print('Mean after {}:'.format(cut))

print(data_mean.loc[cut:].mean())

print('')

print('---------------------------')

print('')

print('Std before {}:'.format(cut))

print(data_mean.loc[:cut].std())

print('')

print('Std after {}:'.format(cut))

print(data_mean.loc[cut:].std())
from statsmodels.tsa.stattools import adfuller



result = adfuller(data_mean['San Francisco'])

print('San Francisco')

print('--------------------------')

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))



print('\n\n')

    

result = adfuller(data_mean['Minneapolis'])

print('Minneapolis')

print('--------------------------')

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))    
import statsmodels.api as sm

print('San Francisco')

_=sm.graphics.tsa.plot_acf(data_mean['San Francisco'])

plt.show()

print('Minneapolis')

_=sm.graphics.tsa.plot_acf(data_mean['Minneapolis'])

plt.show()
import statsmodels.api as sm

print('San Francisco')

_=sm.graphics.tsa.plot_acf(data_mean['San Francisco'], lags=365)

plt.show()

print('Minneapolis')

_=sm.graphics.tsa.plot_acf(data_mean['Minneapolis'], lags=365)

plt.show()
print('San Francisco')

_=sm.graphics.tsa.plot_pacf(data_mean['San Francisco'], lags=30)

plt.show()

print('Minneapolis')

_=sm.graphics.tsa.plot_pacf(data_mean['Minneapolis'], lags=30)

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose as sd

sd_SF = sd(data_mean['San Francisco'], freq=365)

sd_M = sd(data_mean['Minneapolis'], freq=365)



_=plt.figure(figsize=(15,10))

ax1=plt.subplot(311)

_=ax1.plot(sd_SF.trend, label='San Francisco', alpha=0.7)

_=ax1.plot(sd_M.trend, label='Minneapolis', alpha=0.7)

_=plt.legend()

ax2=plt.subplot(312)

_=ax2.plot(sd_SF.seasonal, label='San Francisco', alpha=0.7)

_=ax2.plot(sd_M.seasonal, label='Minneapolis', alpha=0.7)

_=plt.legend()

ax3=plt.subplot(313)

_=ax3.plot(sd_SF.resid, label='San Francisco', alpha=0.7)

_=ax3.plot(sd_M.resid, label='Minneapolis', alpha=0.7)

_=plt.legend()