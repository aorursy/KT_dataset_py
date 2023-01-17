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
df.head()
df.tail()
df2 = df.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df2
df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby('Date').sum()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
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

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered',

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
confirmed = df2.groupby(['Date', 'Country']).sum()[['Confirmed']].reset_index()

deaths = df2.groupby(['Date', 'Country']).sum()[['Deaths']].reset_index()

recovered = df2.groupby(['Date', 'Country']).sum()[['Recovered']].reset_index()
latest_date = confirmed['Date'].max()

latest_date
confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]

deaths = deaths[(deaths['Date']==latest_date)][['Country', 'Deaths']]

recovered = recovered[(recovered['Date']==latest_date)][['Country', 'Recovered']]
all_countries = confirmed['Country'].unique()

print("Number of countries with cases: " + str(len(all_countries)))

print("Countries with cases: ")

for i in all_countries:

    print("    " + str(i))
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

confirmed["iso_alpha"] = confirmed["Country"].map(countries.get)

deaths["iso_alpha"] = deaths["Country"].map(countries.get)

recovered["iso_alpha"] = recovered["Country"].map(countries.get)
plot_data_confirmed = confirmed[["iso_alpha","Confirmed", "Country"]]

plot_data_deaths = deaths[["iso_alpha","Deaths"]]

plot_data_recovered = recovered[["iso_alpha","Recovered"]]
fig = px.scatter_geo(plot_data_confirmed, locations="iso_alpha", color="Country",

                     hover_name="iso_alpha", size="Confirmed",

                     projection="natural earth", title = 'Worldwide Confirmed Cases')

fig.show()
fig = px.scatter_geo(plot_data_deaths, locations="iso_alpha", color="Deaths",

                     hover_name="iso_alpha", size="Deaths",

                     projection="natural earth", title="Worldwide Death Cases")

fig.show()
fig = px.scatter_geo(plot_data_recovered, locations="iso_alpha", color="Recovered",

                     hover_name="iso_alpha", size="Recovered",

                     projection="natural earth", title="Worldwide Recovered Cases")

fig.show()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)