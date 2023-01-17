import datetime

import pandas as pd

import requests

import matplotlib as mpl

import matplotlib.pyplot as plt

import os

import plotly.express as px

import numpy as np



mpl.rcParams['figure.figsize'] = (14,8)

mpl.rcParams['axes.grid'] = False



df = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')

df
df.timestamp = pd.to_datetime(df['timestamp'])

df.info()

df = df.set_index('timestamp').resample("H").mean()

df

# !pip install fbprophet
fig = px.line(df.reset_index(), x = 'timestamp',y = 'value',title = 'NYC_Taxi_Demand')



fig.update_xaxes(

    rangeslider_visible= True,

    rangeselector=dict(

                        buttons = list([

                        dict(count = 1,label = '1y',step='month',stepmode = "backward"),

                        dict(count = 2,label = '3y',step='month',stepmode = "backward"),

                        dict(count = 3,label = '5y',step='month',stepmode = "todate"),

                        dict(step= 'all')

                            ])        

                        )

                   )

fig.show()
from fbprophet import Prophet

taxi_df = df.reset_index()[['timestamp','value']].rename({'timestamp':'ds','value':'y'}, axis='columns')



taxi_df



train = taxi_df[(taxi_df['ds']>='2014-07-01') & (taxi_df['ds']<='2015-01-27')]

test = taxi_df[(taxi_df['ds']>'2015-01-27')]
print(train.shape)

print(test.shape)

test

m = Prophet(changepoint_range=0.95)

m.fit(train)

m.params



future = m.make_future_dataframe(periods=119,freq='H')

future.tail(167)
forecast = m.predict(future)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
results = pd.concat([taxi_df.set_index('ds')['y'],forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]],axis=1)

results
fig1 = m.plot(forecast)

comp = m.plot_components(forecast)

results


results['error'] = results['y'] - results['yhat']

# print("************************")

# print(results)
results['uncertainity'] = results['yhat_upper'] - results['yhat_lower']

results
results[results['error'].abs() > 1.5*results['uncertainity']]

results['anomaly'] = results.apply(lambda x: 'Yes' if (np.abs(x['error']) > 1.5*x['uncertainity']) else 'No', axis=1)

results
fig = px.scatter(results.reset_index(), x = 'ds',y = 'y',color='anomaly',title = 'NYC_Taxi_Demand')



fig.update_xaxes(

    rangeslider_visible= True,

    rangeselector=dict(

                        buttons = list([

                        dict(count = 1,label = '1y',step='month',stepmode = "backward"),

                        dict(count = 2,label = '3y',step='month',stepmode = "backward"),

                        dict(count = 3,label = '5y',step='month',stepmode = "todate"),

                        dict(step= 'all')

                            ])        

                        )

                   )

fig.show()
comp=m.plot_components(forecast)