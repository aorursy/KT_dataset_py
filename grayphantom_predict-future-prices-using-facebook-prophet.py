# import libraries 

import pandas as pd # Import Pandas for data manipulation using dataframes

import numpy as np # Import Numpy for data statistical analysis 

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import random

import seaborn as sns

from fbprophet import Prophet

import plotly.graph_objects as go
# dataframes creation for both training and testing datasets 

avocado_data=pd.read_csv('../input/avocado-prices/avocado.csv')
# Let's view the head of the training dataset

avocado_data.head()
# Let's view the last elements in the training dataset

avocado_data.tail(10)
avocado_data.describe()
avocado_data.info()
avocado_data.isnull().sum()
avocado_data=avocado_data.sort_values('Date')
# Plot date and average price

fig=go.Figure()

fig.add_trace(go.Scatter(x=avocado_data.Date,y=avocado_data.AveragePrice,mode='lines'))

fig.show()
# Plot distribution of the average price

plt.figure(figsize=(10,6))

sns.distplot(avocado_data['AveragePrice'])
# Plot a violin plot of the average price vs. avocado type

sns.violinplot(y='AveragePrice',x='type',data=avocado_data)
# Bar Chart to indicate the number of regions 

sns.set(font_scale=0.7) 

plt.figure(figsize=(16,8))

sns.countplot(x = 'region', data = avocado_data)

plt.xticks(rotation = 45)

plt.show()
# Bar Chart to indicate the count in every year

sns.set(font_scale=1.5) 

plt.figure(figsize=(16,8))

sns.countplot(x = 'year', data = avocado_data)

plt.xticks(rotation = 45)
 # plot the avocado prices vs. regions for conventional avocados

sns.catplot('AveragePrice','region',data=avocado_data[avocado_data['type']=='conventional'],hue='year',height=20)
  # plot the avocado prices vs. regions for organic avocados

sns.catplot('AveragePrice','region',data=avocado_data[avocado_data['type']=='organic'],hue='year',height=20)
data=avocado_data[['Date','AveragePrice']]
data=data.rename(columns={'Date':'ds','AveragePrice':'y'})
data
model=Prophet()

model.fit(data)
# Forcasting into the future

future=model.make_future_dataframe(periods=365)

forecast=model.predict(future)
forecast
fig=model.plot(forecast,xlabel='Date',ylabel='Price',figsize=(16,8))
fig=model.plot_components(forecast,figsize=(12,6))
# Select specific region

region_data=avocado_data[avocado_data['region']=='Chicago']
region_data=region_data.sort_values('Date')
region_data=region_data.rename(columns={'Date':'ds','AveragePrice':'y'})
region_model = Prophet()

region_model.fit(region_data)

future = region_model.make_future_dataframe(periods=365)

forecast = region_model.predict(future)
figure = region_model.plot(forecast, xlabel='Date', ylabel='Price',figsize=(12,8))
figure = region_model.plot_components(forecast,figsize=(12,8))
#Organic avocados

organic_data=avocado_data[avocado_data['type']=='organic']

organic_data=organic_data.sort_values('Date')

organic_data=organic_data.rename(columns={'Date':'ds','AveragePrice':'y'})
organic_model=Prophet()

organic_model.fit(organic_data)

future=organic_model.make_future_dataframe(365)

forecast=organic_model.predict(future)
figure=organic_model.plot(forecast,xlabel="Date",ylabel="Price",figsize=(12,8))
figure=organic_model.plot_components(forecast,figsize=(12,8))
#conventional avocados

conv_data=avocado_data[avocado_data['type']=='conventional']

conv_data=conv_data.sort_values('Date')

conv_data=conv_data.rename(columns={'Date':'ds','AveragePrice':'y'})
conv_model=Prophet()

conv_model.fit(conv_data)

future=conv_model.make_future_dataframe(365)

forecast=conv_model.predict(future)
figure=conv_model.plot(forecast,xlabel="Date",ylabel="Price",figsize=(12,8))
figure=conv_model.plot_components(forecast,figsize=(12,8))