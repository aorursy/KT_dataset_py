import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")

df.head()
df.info()
df['Date']=pd.to_datetime(df['Date'])

df['Date'].dtype
df['Year']=df['Date'].dt.year

df['Month']=df['Date'].dt.month

df['Month']=(df['Year'].astype(str)+df['Month'].astype(str)).astype(int)

df['Week']=df['Date'].dt.week

df['Week']=(df['Year'].astype(str)+df['Week'].astype(str)).astype(int)

df.head()
print ("Date range: {} to {}".format(df['Date'].min(),df['Date'].max()))
df2=df.groupby('Date').agg({'Sales':sum}).reset_index()

df2.rename({'Date':'ds','Sales':'y'},inplace=True,axis=1)

df2.head()
from fbprophet import Prophet
m=Prophet()

m.fit(df2)
future = m.make_future_dataframe(periods=365)

#Periods indicate the number of future days for which we are predicting.

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()
fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
fig2 = m.plot_components(forecast)
from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
playoffs = pd.DataFrame({

  'holiday': 'playoff',

  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',

                        '2010-01-24', '2010-02-07', '2011-01-08',

                        '2013-01-12', '2014-01-12', '2014-01-19',

                        '2014-02-02', '2015-01-11', '2016-01-17',

                        '2016-01-24', '2016-02-07']),

  'lower_window': 0,

  'upper_window': 1,

})



playoffs
df.head()
state_holiday = pd.DataFrame({

  'holiday': 'state_holiday',

  'ds': df[df['StateHoliday']!=0]['Date']

})



school_holiday = pd.DataFrame({

  'holiday': 'school_holiday',

  'ds': df[df['SchoolHoliday']==1]['Date']

})



holidays=pd.concat((state_holiday,school_holiday))

holidays.sample(5)
m = Prophet(holidays=holidays)

forecast = m.fit(df2).predict(future)

fig=m.plot(forecast)
m.plot_components(forecast)