# import libraries 

import pandas as pd # Import Pandas for data manipulation using dataframes

import numpy as np # Import Numpy for data statistical analysis 

import matplotlib.pyplot as plt # Import matplotlib for  data visualisation

import random

import seaborn as sns

from fbprophet import Prophet
# dataframes creation for both training and testing datasets 

avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')
# Let's view the head of the training dataset

avocado_df.head()
# Let's view the last elements in the training dataset

avocado_df.tail()
avocado_df.describe()
avocado_df.info()
avocado_df.isnull().sum()
avocado_df = avocado_df.sort_values('Date')
# Plot date and average price

plt.figure(figsize=(10,10))

plt.plot(avocado_df['Date'],avocado_df['AveragePrice'])
# Plot distribution of the average price

plt.figure(figsize=(10,6))

sns.distplot(avocado_df['AveragePrice'],color='b')
# Plot a violin plot of the average price vs. avocado type

sns.violinplot(y='AveragePrice',x='type',data=avocado_df)
# Bar Chart to indicate the number of regions 



sns.set(font_scale=0.7) 

plt.figure(figsize=[20,8])

sns.countplot(x = 'region', data = avocado_df)

plt.xticks(rotation = 45)

# Bar Chart to indicate the count in every year

sns.set(font_scale=1.5) 

plt.figure(figsize=[16,8])

sns.countplot(x = 'year', data = avocado_df)

plt.xticks(rotation = 45)
 # plot the avocado prices vs. regions for conventional avocados

conventional = sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='conventional'],hue='year',

                           height=15)

  # plot the avocado prices vs. regions for organic avocados

conventional = sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='organic'],hue='year',

                           height=15)

    

avocado_df
avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
avocado_prophet_df
avocado_prophet_df.rename(columns={'Date':'ds','AveragePrice' : 'y'},inplace='true')
avocado_prophet_df
m = Prophet()

m.fit(avocado_prophet_df)
# Forcasting into the future

future = m.make_future_dataframe(periods = 365)

forecast = m.predict(future)
forecast
figure = m.plot(forecast, xlabel='Date', ylabel='Price')


figure2 = m.plot_components(forecast)
# dataframes creation for both training and testing datasets 

avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')
# Select specific region

avocado_df_sample = avocado_df[avocado_df['region']=='West']
avocado_df_sample = avocado_df_sample.sort_values('Date')
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])
avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})
m1 = Prophet()

m1.fit(avocado_df_sample)

# Forcasting into the future

future1 = m1.make_future_dataframe(periods=365)

forecast1 = m1.predict(future)
figure = m1.plot(forecast1, xlabel='Date', ylabel='Price')
figure3 = m.plot_components(forecast)