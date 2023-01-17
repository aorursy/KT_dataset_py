import pandas as pd

import numpy as np

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['ObservationDate'])

df.rename(columns={'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)  # Renaming the column 'Country/Region' to 'Country'



df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)



df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)



df.head(3)
# Creating a dataframe df_mauritius which will contain all information for COVID19 cases in Mauritius

df_mauritius = df.query('Country=="Mauritius"').groupby("ObservationDate")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()  # getting date, confirmed, deaths, recovered

df_mauritius['ObservationDate'] = df_mauritius['ObservationDate'].dt.date  #

df_mauritius['Active'] = df_mauritius['Confirmed'] - df_mauritius['Deaths'] - df_mauritius['Recovered']
df_mauritius
print('The first case was detected on ',df_mauritius['ObservationDate'].iloc[0], '.', sep="")

print('Data last updated: ', df_mauritius['ObservationDate'].iloc[-1])

print('Total Confirmed Cases: ',int(df_mauritius['Confirmed'].max()), sep="")

print('Current Active Cases: ', int(df_mauritius['Active'].max()), sep="")

print('Recovered Cases: ',int(df_mauritius['Recovered'].sum()), sep="")

print('Deaths: ',int(df_mauritius['Deaths'].max()), sep="")
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_mauritius['ObservationDate'], y=df_mauritius['Confirmed'], mode='lines+markers', name='Confirmed', line=dict(color='blue', width=2)))

fig.update_layout(title='Mauritius - Confirmed cases (Line Chart)', yaxis=dict(title='Number of Cases'))

fig.show()
fig = px.bar(x=df_mauritius['ObservationDate'], y=df_mauritius['Confirmed'], text=df_mauritius['Confirmed'])

fig.update_traces(texttemplate='%{text} cases', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(title='Mauritius - Confirmed cases (Bar Chart)', yaxis=dict(title='Number of Cases'), xaxis=dict(title='Date'))

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=df_mauritius['ObservationDate'], y=df_mauritius['Confirmed'], mode='lines+markers', name='Confirmed', line=dict(color='Blue', width=2)))

fig.add_trace(go.Scatter(x=df_mauritius['ObservationDate'], y=df_mauritius['Active'], mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))

fig.add_trace(go.Scatter(x=df_mauritius['ObservationDate'], y=df_mauritius['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))

fig.add_trace(go.Scatter(x=df_mauritius['ObservationDate'], y=df_mauritius['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))



fig.update_layout(title='Mauritius - Confirmed, Deaths, Recovered (Line Chart)', yaxis=dict(title='Number of Cases'),    legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'))

fig.show()
fig = go.Figure()



fig.add_trace(go.Bar(x=df_mauritius['ObservationDate'], y=df_mauritius['Confirmed'], name='Confirmed per day', marker_color='yellow'))

fig.add_trace(go.Bar(x=df_mauritius['ObservationDate'], y=df_mauritius['Deaths'], name='Deaths per day', marker_color='red'))

fig.add_trace(go.Bar(x=df_mauritius['ObservationDate'], y=df_mauritius['Recovered'], name='Recovered per day', marker_color='green'))



fig.update_layout(barmode='group', xaxis_tickangle=-0, legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'))

fig.show()