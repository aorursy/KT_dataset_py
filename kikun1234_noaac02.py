import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet

import matplotlib.pyplot as plt
df = pd.read_excel("../input/noaa-co2/CO2.xlsx")
df = df.drop(columns="trend")
df.head()
df['year'] = df['year'].astype(str)

df['month'] = df['month'].astype(str)

df['day'] = df['day'].astype(str)
df['ds'] = df['year'] + "-" + df['month'] +"-" + df['day']
df.head()
df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds','cycle']]
df = df.rename(columns={"cycle":"y"})
df.head()
m = Prophet(weekly_seasonality=False)

m.add_seasonality(name='monthly', period=12, fourier_order=5)

m.fit(df)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

m.plot(forecast)

plt.title("Time Series Forecasting with Prophet")

plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)