import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)



# Serbia

serbia = df.query('Country=="Serbia"')

serbia.groupby('Date').sum()

confirmed = serbia.groupby('Date').sum()['Confirmed'].reset_index()

deaths = serbia.groupby('Date').sum()['Deaths'].reset_index()

recovered = serbia.groupby('Date').sum()['Recovered'].reset_index()
fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'], 

                         y=confirmed['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['Date'], 

                         y=deaths['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=recovered['Date'], 

                         y=recovered['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Serbia Corona Virus Cases - Confirmed, Deaths, Recovered',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
confirmed = serbia.groupby('Date').sum()['Confirmed'].reset_index()

deaths = serbia.groupby('Date').sum()['Deaths'].reset_index()

recovered = serbia.groupby('Date').sum()['Recovered'].reset_index()

confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=60)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
review = forecast[20:35].rename(columns={'ds': 'date', 'yhat': 'mid', 'yhat_lower': 'min', 'yhat_upper': 'max'}).reset_index()

review[['date', 'min', 'mid', 'max']]