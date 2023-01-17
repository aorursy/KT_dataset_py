import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

import seaborn as sns

from fbprophet import Prophet
avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')
avocado_df.head()
avocado_df.tail(10)
avocado_df.describe()
avocado_df.info()
avocado_df.isnull().sum()
avocado_df = avocado_df.sort_values('Date')
plt.figure(figsize = (10, 10))

plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])
plt.figure(figsize = (10, 6))

sns.distplot(avocado_df['AveragePrice'], color = 'b')
sns.violinplot(y = 'AveragePrice', x = 'type', data = avocado_df)
sns.set(font_scale=0.7) 

plt.figure(figsize=[25,12])

sns.countplot(x = 'region', data = avocado_df)

plt.xticks(rotation = 45)
sns.set(font_scale=1.5) 

plt.figure(figsize=[25,12])

sns.countplot(x = 'year', data = avocado_df)

plt.xticks(rotation = 45)
conventional = sns.catplot('AveragePrice', 'region', data = avocado_df[avocado_df['type']=='conventional'], 

                           hue = 'year',

                           height = 20)
organic = sns.catplot('AveragePrice', 'region', data = avocado_df[avocado_df['type']=='organic'],

                      hue = 'year',

                      height = 20)
avocado_df
avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
avocado_prophet_df
avocado_prophet_df = avocado_prophet_df.rename(columns = {'Date':'ds', 'AveragePrice':'y'})
avocado_prophet_df
m = Prophet()

m.fit(avocado_prophet_df)
# Forcasting into the future

future = m.make_future_dataframe(periods = 365)

forecast = m.predict(future)
figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Price')
figure2 = m.plot_components(forecast)
avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')
# Select specific region

avocado_df_sample = avocado_df[avocado_df['region']=='West']
avocado_df_sample = avocado_df_sample.sort_values('Date')
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])
avocado_df_sample = avocado_df_sample.rename(columns = {'Date':'ds', 'AveragePrice':'y'})
m = Prophet()

m.fit(avocado_df_sample)

# Forcasting into the future

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)
figure = m.plot(forecast, xlabel='Date', ylabel='Price')
figure3 = m.plot_components(forecast)