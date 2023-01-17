import pandas as pd

import numpy

import seaborn as sns

import matplotlib.pyplot as plt

import fbprophet

import plotly

import warnings

import plotly.graph_objs as go 



from datetime import datetime



plt.style.use("ggplot")

warnings.filterwarnings("ignore")



itemcat=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv',encoding='utf-8',low_memory=False)

item=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv',encoding='utf-8',low_memory=False)

sales=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',encoding='ISO-8859-1',low_memory=False)

sample=pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv',encoding='utf-8',low_memory=False)

shop=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv',encoding='utf-8',low_memory=False)

test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv',encoding='utf-8',low_memory=False)
itemcat.head()
item.head()
sales.head()
sample.head()
shop.head()
test.head()
df = sales.iloc[:,[0,4]]

df.columns = ['Date','Price']

df.Date = pd.to_datetime(df.Date)

df = df.groupby(by=['Date'],sort=True,as_index=False).count()

df.tail()
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

fbp = fbprophet.Prophet()

fbp.fit(df)
df_forecast = fbp.make_future_dataframe(periods=12,freq='M')

df_forecast = fbp.predict(df_forecast)

fbp.plot(df_forecast, xlabel = 'Date', ylabel = 'Price')

plt.title('Next Month Sales')
holiday= pd.DataFrame({

  'ds': pd.to_datetime(['2014-09-01', '2015-07-01']),

  'holiday': 'start',

  'lower_window': 0,

  'upper_window': 3,

})

holiday.head()
fbp = fbprophet.Prophet(holidays=holiday)

fbp.fit(df)

df_forecast = fbp.make_future_dataframe(periods=24,freq='M')

df_forecast2 = fbp.predict(df_forecast)

fbp.plot(df_forecast2, xlabel = 'Date', ylabel = 'Price')

plt.title('Next Month Sales')
fbp.plot_components(df_forecast2)
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

df['y'] = df['y'] / 1e9

df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)

df_prophet.fit(df)
df_forecast = df_prophet.make_future_dataframe(periods=365 * 2, freq='D')

df_forecast = df_prophet.predict(df_forecast)

df_prophet.plot(df_forecast, xlabel = 'Date', ylabel = 'Price')

plt.title('Next Month Sales')