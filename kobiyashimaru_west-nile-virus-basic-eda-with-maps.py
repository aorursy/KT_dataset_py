!unzip -oq ../input/predict-west-nile-virus/spray.csv.zip
!unzip -oq ../input/predict-west-nile-virus/train.csv.zip
!unzip -oq ../input/predict-west-nile-virus/weather.csv.zip
import os
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

MAPBOX_TOKEN = ''
for files in os.listdir():
    print(files) if files.endswith('csv') else None
train = pd.read_csv('train.csv')
spray = pd.read_csv('spray.csv')
weather = pd.read_csv('weather.csv')
train.columns
train.describe()
train.WnvPresent.value_counts()
# If maps are not being displayed, please re-run the cell
px.set_mapbox_access_token(MAPBOX_TOKEN)
fig = px.scatter_mapbox(train, lat = 'Latitude', lon  = 'Longitude',
                        size_max=15, zoom = 10)

fig.update_layout(title = 'Traps',
    autosize=False,
    width=500,
    height=700,)

fig.show()
spray.describe()
px.set_mapbox_access_token(MAPBOX_TOKEN)

fig = px.scatter_mapbox(spray, lat = 'Latitude', lon  = 'Longitude',
                     animation_frame="Date",
                        size_max=15, zoom = 9)

fig.update_layout(
    title="Spray day-wise",
        width=500,
    height=700,
)

fig.show()
mosquito_count = train.groupby(['Address'], as_index = False)[['NumMosquitos']].sum()
areas = train.groupby(['Address'], as_index = False)[['Latitude','Longitude']].median()
wnv = train.groupby(['Address'], as_index = False)[['WnvPresent']].sum() 
# sum() because it has either 0 or 1 values. adding ones will give us total cases in an area.
mosquito_areas_wnv = pd.concat([mosquito_count,areas, wnv], axis = 1)
mosquito_areas_wnv.drop('Address', axis = 1, inplace = True)
fig = px.scatter_mapbox(mosquito_areas_wnv, lat = 'Latitude', lon  = 'Longitude', color = 'WnvPresent',
                        size = 'NumMosquitos', color_continuous_scale=px.colors.cyclical.IceFire,
                        hover_data = ['NumMosquitos', 'WnvPresent'],
                       zoom = 9)
fig.show()

fig = px.scatter_mapbox(spray, lat = 'Latitude', lon  = 'Longitude',#animation_frame = 'Date',
                        size_max=15, zoom = 9,color_discrete_sequence=["palegoldenrod"],  opacity = 0.5)

fig2 = px.scatter_mapbox(mosquito_areas_wnv, lat = 'Latitude', lon  = 'Longitude', color = 'WnvPresent',
                        size = 'NumMosquitos', color_continuous_scale=px.colors.cyclical.IceFire,
                        hover_data = ['NumMosquitos', 'WnvPresent'],
                       zoom = 9)

fig.add_trace(fig2.data[0],)

fig.update_layout( title = 'Spray - West Nile Virus and Mosquito clusters',
                width=500,
    height=700,
)
fig2 = px.scatter_mapbox(mosquito_areas_wnv, lat = 'Latitude', lon  = 'Longitude', color = 'WnvPresent',
                        size = 'NumMosquitos', color_continuous_scale=px.colors.cyclical.IceFire,
                        hover_data = ['NumMosquitos', 'WnvPresent'],
                       zoom = 9)

fig3 = px.scatter_mapbox(train, lat = 'Latitude', lon  = 'Longitude',
                        size_max=15, zoom = 10, color_discrete_sequence = ['lemonchiffon'])

#below is one way to plot multiple graphs on the same plot. 
#print figure object as is to see the elements inside
fig2.add_trace(fig3.data[0]) 

fig2.update_layout(mapbox_style='dark')

fig2.update_layout( title = 'Traps - West Nile Virus and Mosquito clusters',
                width=500,
    height=700,)

species_vs_virus = train[['Species', 'WnvPresent']].groupby('Species', as_index = False).sum()
species_vs_virus
fig = px.bar(species_vs_virus, x = 'Species', y = 'WnvPresent')
fig.update_layout(
    title="West Nile Virus count vs Species",
    xaxis_title="Species",
    yaxis_title="West Nile Virus Present",)
fig.show()

weather.head()
weather['Tavg'].unique()
weather[weather['Tavg']=='M']
# Only 11 rows, so drop these for now.
weather.drop(weather[weather['Tavg']=='M'].index, axis = 0, inplace = True)
weather.reset_index(drop = True)
weather.columns
weather['Tavg'] = weather['Tavg'].astype(int) 
weather_imp = weather.groupby(['Date'], as_index = False)[['Tavg']].mean()
weather_imp
mosquitos_date_wise = train.groupby(['Date'], as_index = False)[['NumMosquitos']].sum()
wnv_date_wise = train.groupby(['Date'], as_index = False)[['WnvPresent']].sum()
wnv_mosquitos_dw = pd.merge(mosquitos_date_wise,wnv_date_wise, on = 'Date')
weather_df = pd.merge(wnv_mosquitos_dw, weather_imp)
weather_df
fig = px.scatter(weather_df, x="Tavg", y="NumMosquitos",
                 size='WnvPresent')

fig.update_layout(
    title="Mosquitos vs Average temperature",
    xaxis_title="Average Temperature in Fahrenheit",
    yaxis_title="Number of Mosquitos",)
fig.show()