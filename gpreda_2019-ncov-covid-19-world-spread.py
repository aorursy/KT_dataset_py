import pandas as pd

import numpy as np

import os

import datetime as dt

from datetime import date

# plotly express for visualization

import plotly.express as px

data_df = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')
dt_string = dt.datetime.now().strftime("%d/%m/%Y")

print(f"Kernel last updated: {dt_string}")

mindate = min(data_df['Date'])

maxdate = max(data_df['Date'])

print(f"Date min/max: {mindate}, {maxdate}")

data_df['D'] = data_df['Date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))

data_df['Days'] = data_df['D'].apply(lambda x: (x - dt.datetime.strptime('2020-01-21', "%Y-%m-%d")).days)

data_df.loc[data_df.Confirmed.isna(), 'Confirmed'] = 0

data_df['Size'] = np.round(5 * np.sqrt(data_df['Confirmed']),0)

max_confirmed = max(data_df['Confirmed'])

min_confirmed = min(data_df['Confirmed'])

hover_text = []

for index, row in data_df.iterrows():

    hover_text.append(('Date: {}<br>'+

                       'Country/Region: {}<br>'+

                       'Province/State: {}<br>'+

                      'Confirmed: {}<br>'+

                      'Recovered: {}<br>'+

                      'Deaths: {}<br>').format(row['Date'], 

                                            row['Country/Region'],

                                            row['Province/State'],

                                            row['Confirmed'],

                                            row['Recovered'],

                                            row['Deaths']))

data_df['hover_text'] = hover_text

fig = px.scatter_geo(data_df, lon='Longitude', lat='Latitude', color="Confirmed",

                     hover_name="hover_text", size="Size",

                     animation_frame="Days",

                     projection="natural earth",

                    range_color =[min_confirmed,max_confirmed],

                    width=700, height=525, size_max=50)

fig.update_geos(   

    showcoastlines=True, coastlinecolor="DarkBlue",

    showland=True, landcolor="LightGrey",

    showocean=True, oceancolor="LightBlue",

    showlakes=True, lakecolor="Blue",

    showrivers=True, rivercolor="Blue",

    showcountries=True, countrycolor="DarkBlue"

)

fig.update_layout(title = 'COVID-19 Confirmed cases per Country/Province/State<br>(hover for details)')

fig.show()
data_df.loc[data_df.Deaths.isna(), 'Deaths'] = 0

data_df['Size'] = np.round(5 * np.sqrt(data_df['Deaths']),0)

max_deaths = max(data_df['Deaths'])

min_deaths = min(data_df['Deaths'])

hover_text = []

for index, row in data_df.iterrows():

    hover_text.append(('Date: {}<br>'+

                       'Country/Region: {}<br>'+

                       'Province/State: {}<br>'+

                      'Confirmed: {}<br>'+

                      'Recovered: {}<br>'+

                      'Deaths: {}<br>').format(row['Date'], 

                                            row['Country/Region'],

                                            row['Province/State'],

                                            row['Confirmed'],

                                            row['Recovered'],

                                            row['Deaths']))

data_df['hover_text'] = hover_text

fig = px.scatter_geo(data_df, lon='Longitude', lat='Latitude', color="Deaths",

                     hover_name="hover_text", size="Size",

                     animation_frame="Days",

                     projection="natural earth",

                     range_color =[min_deaths,max_deaths],

                     width=700, height=525, size_max=50)

fig.update_geos(   

    showcoastlines=True, coastlinecolor="DarkBlue",

    showland=True, landcolor="LightGrey",

    showocean=True, oceancolor="LightBlue",

    showlakes=True, lakecolor="Blue",

    showrivers=True, rivercolor="Blue",

    showcountries=True, countrycolor="DarkBlue"

)

fig.update_layout(title = 'COVID-19 Deaths per Country/Province/State<br>(hover for details)')

fig.show()