# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install chart_studio
import geopandas

import chart_studio.plotly as py

import plotly.tools as tls

import plotly.graph_objs as go

import plotly

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

plotly.offline.init_notebook_mode(connected=True)

import folium
airport_col = ['ID', 'Name', 'City', 'Country','IATA', 'ICAO', 'Latitude', 'Longitude', 'Alt', 'Timezone', 'DST', 'Tz database time zone', 'type', 'source']
df = pd.read_csv("../input/online-activity2/world_airports.csv", header = None, names = airport_col, index_col = 0)
df.head()
df.dropna(inplace=True)

df = df.reset_index()
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")



def state(coord):

    location = geolocator.reverse(coord, exactly_one=True)

    address = location.raw['address']

    state = address.get('state', '')

    return state
df_India = df[df['Country']=='India']

df_India = df_India.reset_index()
df_India.head()
df_India['coordinate'] = df_India['Latitude'].astype(str) + "," + df_India['Longitude'].astype(str)

df_India['State'] = df_India['coordinate'].apply(lambda x:state(x))
df_India.head()
import json

import requests
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
url = "https://raw.githubusercontent.com/covid19india/covid19india-react/master/public/maps/india.json"



m = folium.Map(

location=[df_India['Latitude'].mean(), df_India['Longitude'].mean()],

zoom_start= 5

)





folium.TopoJson(

json.loads(requests.get(url).text),

'objects.states',

name='topojson'

).add_to(m)



folium.LayerControl().add_to(m)



for i in range(0,len(df_India)):

    index = (i)%len(colors)

    icon = folium.Icon(color=colors[index], icon="ok")

    folium.Marker(location = [df_India.iloc[i]['Latitude'], df_India.iloc[i]['Longitude']], popup= df_India.iloc[i]['Name'],icon=icon).add_to(m)



m
import geopandas as gpd
groupby_Country = df.groupby(['Country'])
Unique_Countries = df['Country'].unique()
Airport_count_based_on_Country = []



for country in Unique_Countries:

    Airport_count_based_on_Country.append([country,groupby_Country.get_group(country).shape[0]])
Airport_count_based_on_Country = sorted(Airport_count_based_on_Country, key = lambda x: x[1], reverse=True)
mapit = None

coordinates_all = []



for country,count in Airport_count_based_on_Country[0:5]:

    for i in range(df_India.shape[0]):

        temp = groupby_Country.get_group(country)

        temp = temp.reset_index()

        mapit = folium.Map( location=[temp['Latitude'][i], temp['Longitude'][i]], color='crimson',

      fill=True,

      fill_color='crimson',zoom_start=4)

        coordinates_all.append([temp['Latitude'][i], temp['Longitude'][i]])
for point in range(len(coordinates_all)):

    folium.Circle(coordinates_all[point], radius=10, color='crimson',

      fill=True,

      fill_color='crimson'

).add_to(mapit)

mapit
Country_Airport_df = pd.DataFrame(Airport_count_based_on_Country, columns=['Country','Airport_Count'])

Country_Airport_df.head()
world_choropelth = folium.Map(location=[0, 0], tiles='Mapbox Bright',zoom_start=1)



world_choropelth.choropleth(

    geo_data="../input/world-countries/world-countries.json",

    data=Country_Airport_df,

    columns=['Country','Airport_Count'],

    key_on='feature.properties.name',

    fill_color='YlOrRd',

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Airport Count across the world')



folium.LayerControl().add_to(world_choropelth)

# display map

world_choropelth
India_states = df_India.groupby('State')['Name'].count().reset_index()

India_states.columns = ['state','count']

India_states = India_states.sort_values(by='count',ascending=False)
url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"

js = json.loads(requests.get(url).text)

for i in range(len(js['features'])):

    if js['features'][i]['properties']['NAME_1'] in India_states['state'].tolist():

        js['features'][i]['properties']['count'] = str(India_states[India_states['state']== js['features'][i]['properties']['NAME_1']]['count'].iloc[0])

    else:

        js['features'][i]['properties']['count'] = 0





m = folium.Map(location=[21, 78], zoom_start=5)



choropleth = folium.Choropleth(

     geo_data = js,

     name = 'choropleth',

     data = India_states,

     columns = ['state', 'count'],

     key_on = 'feature.properties.NAME_1',

     fill_color = 'YlOrBr',

     fill_opacity = 0.7,

     line_opacity = 0.2,

     nan_fill_color='black',

     legend_name = 'Airport density across states of India'

    ).add_to(m)



choropleth.geojson.add_child(

    folium.features.GeoJsonTooltip(['NAME_1', 'count'],labels=False)

)



folium.LayerControl().add_to(m)



m
groupby_timezone = df.groupby('Tz database time zone')[['Name','Latitude','Longitude']].mean().reset_index()

groupby_timezone.head()
from folium import plugins
heat_map = folium.Map([41.8781, -87.6298], zoom_start=11)

timezone = groupby_timezone[['Latitude','Longitude']].values

heat_map.add_children(plugins.HeatMap(timezone, radius=15))

heat_map