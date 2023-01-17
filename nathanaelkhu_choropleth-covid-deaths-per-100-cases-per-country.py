import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_covid = pd.read_csv("../input/corona-virus-report/country_wise_latest.csv")

df_covid.head()
df_covid.drop(["Confirmed","Deaths","Recovered","Active",	"New cases",	"New deaths",	"New recovered","Recovered / 100 Cases",	"Deaths / 100 Recovered",	"Confirmed last week","1 week change",	"1 week % increase",	"WHO Region"],axis=1)
import folium

import json
world_geo = r'../input/world-map/world_countries.json' # geojson file

world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
folium.Choropleth(

    geo_data=world_geo,

    data=df_covid,

    columns=['Country/Region', 'Deaths / 100 Cases'],

    key_on='feature.properties.name',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='COVID Deaths per 100 Cases per Country'

).add_to(world_map)

world_map