import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random 

import seaborn as sns 

from fbprophet import Prophet 

data = pd.read_csv('../input/crude-oil-stock-price/crudeoil.csv', delimiter=',')
data.head()
plt.figure(figsize=(10,10))

sns.heatmap(data.isnull(), cbar = False, cmap = 'YlGnBu')
data.isna().any()
data = data.dropna()
data.isna().any()
data.index = pd.DatetimeIndex(data.Date)
# ID Case Number Date Block IUCR Primary Type Description Location Description Arrest Domestic Beat District Ward Community Area FBI Code X Coordinate Y Coordinate Year Updated On Latitude Longitude Location

data.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True, axis=1)
data
data_final = data.rename(columns={'Date ':'ds', "Adj Close":'y'})
data_final = data.reset_index()
data_final
data_final.columns = ["Date", "Count"]
data_final
data_final = data_final.rename(columns={'Date':'ds', 'Count':'y'})
data_final
m = Prophet()

m.fit(data_final)
future = m.make_future_dataframe(periods=720)#for 365 days

forecast = m.predict(future)
forecast
figure = m.plot(forecast, xlabel='Date', ylabel='Adj Close')
figure3 = m.plot_components(forecast)