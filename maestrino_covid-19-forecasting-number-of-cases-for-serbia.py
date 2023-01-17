import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import plotly.express as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df.head()
df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df_serbia = df.query('Country=="Serbia"').groupby("Date")

df_serbia[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
confirmed = df_serbia.sum()['Confirmed'].reset_index()

deaths = df_serbia.sum()['Deaths'].reset_index()

recovered = df_serbia.sum()['Recovered'].reset_index()
fig = go.Figure()

fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['Confirmed'],

                name='Confirmed',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Deaths'],

                name='Deaths',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=recovered['Date'],

                y=recovered['Recovered'],

                name='Recovered',

                marker_color='Green'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Bar Chart)',

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

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])

confirmed.head()
model = Prophet(interval_width=0.95)

model.fit(confirmed)

future = model.make_future_dataframe(periods=7)

future_confirmed = future.copy() # for non-baseline predictions later on

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
confirmed_forecast_plot = model.plot(forecast)
figure_test = model.plot_components(forecast)
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
model = Prophet(interval_width=0.95)

model.fit(deaths)

future = model.make_future_dataframe(periods=7)

future_deaths = future.copy() # for non-baseline predictions later on

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
deaths_forecast_plot = model.plot(forecast)
figure_test = model.plot_components(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
model = Prophet(interval_width=0.95)

model.fit(recovered)

future = model.make_future_dataframe(periods=7)

future_recovered = future.copy() # for non-baseline predictions later on

future.tail(7)
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
recovered_forecast_plot = model.plot(forecast)
figure_test = model.plot_components(forecast)