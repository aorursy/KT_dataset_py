#import libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet
# Checking out cwd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#importing the dataset into a dataframe
avocado_df = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')
avocado_df
# Let's view the head of the training dataset
avocado_df.head()
# Let's view the last elements in the training dataset
avocado_df.tail(20)
#setting dates in a chronological order
avocado_df = avocado_df.sort_values("Date")
plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])
avocado_df
#bar plot representation of couting elements by region
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)
#bar plot to indicate sales by year
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)
avocado_prophet_df = avocado_df[['Date', 'AveragePrice']] 
avocado_prophet_df
#ranaming our columns for facebook prophet to operate
avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})
avocado_prophet_df
#instantiating our Prophet object
m = Prophet()
m.fit(avocado_prophet_df)
#forecasting the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast
#visualizing future results
figure = m.plot(forecast, xlabel='Date', ylabel='Price')
#expected trend in the future
figure3 = m.plot_components(forecast)
#importing the dataset into a dataframe 
avocado_df = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')
avocado_df
#considering only the 'west' region and splitting the dataset
avocado_df_sample = avocado_df[avocado_df['region']=='West']
avocado_df_sample
#sorting the dataframe by dates
avocado_df_sample = avocado_df_sample.sort_values("Date")
avocado_df_sample
#visualising trend of prices across a specific region only
plt.figure(figsize=(10,10))
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])
#renaming the columns into 'ds' and 'y' format for facebook prophet,
#formatting in 'M' for implementation
avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})
#using facebook prophet to predict the future
m = Prophet()
m.fit(avocado_df_sample)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
#visualizing future results
figure = m.plot(forecast, xlabel='Date', ylabel='Price')
#expected trend in the future
figure3 = m.plot_components(forecast)