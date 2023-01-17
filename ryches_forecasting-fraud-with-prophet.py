!pip install -U fbprophet

from fbprophet import Prophet
import numpy as np

import pandas as pd

import datetime

import math

import gc

pd.plotting.register_matplotlib_converters()



PATH = '../input/'

START_DATE = '2017-12-01'
# Load data

train = pd.read_csv(PATH + 'train_transaction.csv')

test = pd.read_csv(PATH + 'test_transaction.csv')



df = pd.concat([train, test], axis = 0, sort = False)

del train, test

gc.collect()
# Preprocess date column

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))



print(df['TransactionDT'].head())

print(df['TransactionDT'].tail())
tm_df = df.groupby(by=df['TransactionDT'].dt.date)["isFraud"].count()
tm_df = tm_df.reset_index()
n_train = 182

n_test = 365 - n_train
tm_df.columns = ["ds", "y"]

tm_df
tm_df.iloc[:n_train].plot(x = "ds", y = "y")
m = Prophet()

m.fit(tm_df.iloc[:n_train])
future = m.make_future_dataframe(periods=n_test)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)

forecast
fig2 = m.plot_components(forecast)

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
m = Prophet(changepoint_prior_scale=0.3)

forecast = m.fit(tm_df.iloc[:n_train]).predict(future)

fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
#let's pick the 3 biggest values and use them as changepoints

changepoints = tm_df.iloc[:n_train].nlargest(3, "y")["ds"]
m = Prophet(changepoints=changepoints)

forecast = m.fit(tm_df.iloc[:n_train]).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
m = Prophet()

m.add_country_holidays(country_name='US')

forecast = m.fit(tm_df.iloc[:n_train]).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
manual_holidays = pd.DataFrame({

  'holiday': 'manual',

  'ds': pd.to_datetime(['2018-03-3']),

  'lower_window': -1,

  'upper_window': 1,

})



m = Prophet(holidays=manual_holidays, changepoint_prior_scale=.05)

m.add_country_holidays(country_name='US')

forecast = m.fit(tm_df.iloc[:n_train]).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
train_std = tm_df.iloc[:n_train]["y"].std()

train_mean = tm_df.iloc[:n_train]["y"].mean()
tm_df.loc[(tm_df["y"] > (train_mean + 2.5*train_std)), "y"] = np.nan
m = Prophet()

forecast = m.fit(tm_df.iloc[:n_train]).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
tm_df = df.groupby(by=df['TransactionDT'].dt.date)["isFraud"].mean()

tm_df = tm_df.reset_index()

tm_df.columns = ["ds", "y"]
m = Prophet(changepoint_prior_scale=.05)

forecast = m.fit(tm_df.iloc[:n_train]).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)