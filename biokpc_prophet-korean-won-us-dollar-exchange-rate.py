%matplotlib inline

import pandas as pd

import numpy as np

from fbprophet import Prophet



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/DEXKOUS.csv")

df = df.rename(columns={'DATE': 'ds', 'DEXKOUS': 'y'}) # DEXKOUS is value of Korean won regarding to 1 US dollor on each date

df.head(100)
# In row 14 and 84, the value of y is '.'. So I delete these row, by using codes

df=df[df['y'] != '.']

df.head(100)
# Python

m = Prophet()

m.fit(df);

future = m.make_future_dataframe(periods=60)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)

m.plot(forecast);

m.plot_components(forecast);

future = m.make_future_dataframe(periods=1260)

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)

m.plot(forecast);
df.loc[(df['ds'] > '2008-01-01') & (df['ds'] < '2009-09-01'), 'y'] = None

model = Prophet().fit(df)

model.plot(model.predict(future));