import numpy as np 

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import folium

import seaborn as sns

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('/kaggle/input/coronavirus-2019ncov/covid-19-all.csv')
df['Date'] = pd.to_datetime(df['Date'])

df['Week'] = df['Date'].dt.to_period('W')

df[['Confirmed','Recovered','Deaths']] = df[['Confirmed','Recovered','Deaths']].fillna(0).astype(int)

df['Still Infected'] = df['Confirmed'] - df['Recovered'] - df['Deaths']

df["Country/Region"].replace({"Mainland China": "China"}, inplace=True)

countries_affected = df['Country/Region'].unique().tolist()

print("\nTotal countries effected by Corona virus: ",len(countries_affected))
print('Countries with Confirmed Corona Virus Cases:')

df_list= df['Country/Region'].unique().tolist()

print ('[%s]' % ', '.join(map(str, df_list)))

recent_date = df['Date'].max()

latest_entry = df[df['Date'] >= recent_date]

df_confirmed_contries = latest_entry.groupby('Country/Region')['Confirmed'].sum().reset_index().sort_values('Confirmed',ascending = False).head(10)

fig = px.bar(df_confirmed_contries, x='Country/Region', y='Confirmed', color='Confirmed', height=600)

fig.show()
df_recovered_contries = latest_entry.groupby('Country/Region')['Recovered'].sum().reset_index().sort_values('Recovered',ascending = False).head(10)

fig = px.bar(df_recovered_contries, x='Country/Region', y='Recovered', color='Recovered', height=600)

fig.show()
df_deaths_contries = latest_entry.groupby('Country/Region')['Deaths'].sum().reset_index().sort_values('Deaths',ascending = False).head(10)

fig = px.bar(df_deaths_contries, x='Country/Region', y='Deaths', color='Deaths', height=600)

fig.show()
overall_stats = latest_entry.agg({'Still Infected':'sum','Recovered':'sum','Deaths':'sum'}).reset_index()

overall_stats.columns = ['Status','Count']

pie = go.Pie(labels=overall_stats['Status'], values=overall_stats['Count'], marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Global Corona Virus Cases')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
df_cleaned = latest_entry[(latest_entry['Latitude'].notnull()) & (latest_entry['Longitude'].notnull())]

df_cleaned['Confirmed'] = df_cleaned['Confirmed'].fillna(0).astype(int)

df_cleaned['Recovered'] = df_cleaned['Recovered'].fillna(0).astype(int)

df_cleaned['Deaths'] = df_cleaned['Deaths'].fillna(0).astype(int)

df_confirmed = df_cleaned[(df_cleaned['Confirmed'].notnull()) & (df_cleaned['Confirmed'] != 0)]

df_recovered = df_cleaned[(df_cleaned['Recovered'].notnull()) & (df_cleaned['Recovered'] != 0)]

df_infected = df_cleaned[(df_cleaned['Still Infected'].notnull()) & (df_cleaned['Still Infected'] != 0)]

df_deaths = df_cleaned[(df_cleaned['Deaths'].notnull()) & (df_cleaned['Deaths'] != 0)]
confirmed_cases_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(df_confirmed['Latitude'], df_confirmed['Longitude'], df_confirmed['Confirmed'], df_confirmed['Country/Region']):

    folium.CircleMarker([lat, lon], radius=3,popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>' '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),color='blue', fill_color='blue',fill_opacity=0.7 ).add_to(confirmed_cases_map)

confirmed_cases_map
recovered_cases_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(df_recovered['Latitude'], df_recovered['Longitude'], df_recovered['Recovered'], df_recovered['Country/Region']):

    folium.CircleMarker([lat, lon], radius=3,popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>' '<strong>recovered Cases</strong>: ' + str(value) + '<br>'),color='green', fill_color='green',fill_opacity=0.7 ).add_to(recovered_cases_map)

recovered_cases_map
death_cases_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')

for lat, lon, value, name in zip(df_deaths['Latitude'], df_deaths['Longitude'], df_deaths['Deaths'], df_deaths['Country/Region']):

    folium.CircleMarker([lat, lon], radius=3,popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>' '<strong>Death Cases</strong>: ' + str(value) + '<br>'),color='red', fill_color='red',fill_opacity=0.7 ).add_to(death_cases_map)

death_cases_map
fig = px.treemap(df_confirmed.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Confirmed",

                 title='Confirmed Cases by Country',

                 color_discrete_sequence = px.colors.qualitative.G10)

fig.show()



fig = px.treemap(df_recovered.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Recovered", 

                 title='Recovered Cases by Country',

                 color_discrete_sequence = px.colors.qualitative.Vivid)

fig.show()
fig = px.treemap(df_infected.sort_values(by='Still Infected', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Still Infected", 

                 title='Infected Cases by Country',

                 color_discrete_sequence = px.colors.qualitative.Safe)

fig.show()



fig = px.treemap(df_deaths.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values="Deaths", 

                 title='Death Cases by Country',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()
def country(row):

    if row['Country/Region'] == 'US':

        row['Country'] = 'US'

    else:

        row['Country'] = 'Rest of the World'

    return row['Country']



df['Country'] = df.apply(country,axis = 1)
latest_entry = df[df['Date'] >= recent_date]

df_stats = df.groupby(['Week','Country'])[['Confirmed','Recovered','Deaths']].sum().reset_index()

df_stats['Week'] = df_stats['Week'].astype(str)

df_stats['Week'] = df_stats[df_stats['Week'] != '2020-07-27/2020-08-02']



fig = px.line(df_stats, x="Week", y="Confirmed", color='Country',title='Corona Virus Confirmed Count by Week')

fig.show()



fig = px.line(df_stats, x="Week", y="Recovered", color='Country',title='Corona Virus Recovered Count by Week')

fig.show()



fig = px.line(df_stats, x="Week", y="Deaths", color='Country',title='Corona Virus Death Count by Week')

fig.show()
latest_entry_india = latest_entry[latest_entry['Country/Region'] =='India']

latest_entry_india = latest_entry_india.melt(id_vars="Date", value_vars=['Still Infected','Recovered','Deaths'])

fig = px.treemap(latest_entry_india, path=["variable"], values="value", height=200)

fig.show()
latest_entry_US = latest_entry[latest_entry['Country/Region'] =='US']

fig = px.treemap(latest_entry_US.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["Province/State"], values="Confirmed",

                 title='Confirmed Cases in US',

                 color_discrete_sequence = px.colors.qualitative.G10)

fig.show()



# fig = px.treemap(latest_entry_US.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 

#                  path=["Province/State"], values="Recovered", 

#                  title='Recovered Cases in US',

#                  color_discrete_sequence = px.colors.qualitative.Vivid)

# fig.show()
fig = px.treemap(latest_entry_US.sort_values(by='Still Infected', ascending=False).reset_index(drop=True), 

                 path=["Province/State"], values="Still Infected", 

                 title='Infected Cases in US',

                 color_discrete_sequence = px.colors.qualitative.Safe)

fig.show()



fig = px.treemap(latest_entry_US.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["Province/State"], values="Deaths", 

                 title='Death Cases in US',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()