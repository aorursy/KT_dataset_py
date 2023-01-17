import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns # for plot visualization
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import os
print(os.listdir("../input"))
weather_df = pd.read_csv('../input/historical-weather-data-for-indian-cities/pune.csv', parse_dates=['date_time'], index_col='date_time')
pd.set_option('display.max_columns', 5000)
weather_df.head()
weather_df = weather_df.loc[:,['tempC', 'sunHour', 'precipMM', 'pressure']]
print(f'dataset shape (rows, columns) - {weather_df.shape}')
weather_df.head()
weather_df.dtypes, weather_df.index.dtype
weather_df.describe()
weather_df.index = pd.to_datetime(weather_df.index)
weather_df.index
weather_df.isnull().count()
weather_df.ffill(inplace=True)
weather_df[weather_df.isnull()].count()
weather_condition = (weather_df.sunHour.value_counts()/(weather_df.sunHour.value_counts().sum()))*100
weather_condition.plot.bar(figsize=(16,9))
plt.xlabel('Weather Conditions')
plt.ylabel('Percent')
weather_df.plot(subplots=True, figsize=(20,12))

weather_df['2019':'2020'].resample('D').fillna(method='pad').plot(subplots=True, figsize=(20,12))
weather_df = weather_df.loc[:,['tempC']]

train_df = weather_df['2009':'2017'].resample('M').mean().fillna(method='pad')
test_df = weather_df['2017':'2020'].resample('M').mean().fillna(method='pad')

train_df.values

#train_df.tempC.diff().values

#train_df.tempC.diff().diff().values

# Original Series
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':360})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(train_df.values); 
axes[0, 0].set_title('Original Series')
plot_acf(train_df.values, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(train_df.tempC.diff().values); 
axes[1, 0].set_title('1st Order Differencing')
plot_acf(train_df.diff().dropna().values,ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(train_df.tempC.diff().diff().values); 
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(train_df.diff().diff().dropna().values,ax=axes[2, 1])

plt.xticks(rotation='vertical')
plt.show()

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(train_df.diff().values); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(train_df.diff().dropna().values, ax=axes[1])

plt.show()
