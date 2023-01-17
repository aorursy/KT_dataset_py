import numpy as np 

import pandas as pd 

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/dubai-communities/top_100_dubai_communities_by_population.csv")

print(df.shape)

df.head()
df.isnull().sum()
from plotly.offline import iplot, init_notebook_mode

barplot = go.Bar(

                x = df.community,

                y = df.population,

                name = "citations",

                marker = dict(color = 'rgba(255, 174, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.community)



data = [barplot]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df_pop = df[["community", "population"]].iloc[:10]

pop_list = df_pop.population

labels = df_pop.community

fig = {

  "data": [

    {

      "values": pop_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "community",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Top 10 Community Population",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Population",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
import folium

init_lat = df.latitude.mean()

init_long = df.longitude.mean()

def generateBaseMap(default_location=[init_lat, init_long], default_zoom_start=11):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map
map = generateBaseMap()

for i, point in enumerate(zip(df.latitude, df.longitude)):

    folium.Marker((point),popup=df.community.iloc[i]).add_to(map)

map
from folium.plugins import HeatMap

base_map = generateBaseMap(default_zoom_start=10)

heatmap = HeatMap(data=df[['latitude', 'longitude', 'population']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8)

heatmap.add_to(base_map)

base_map
heatmap.add_to(map)

map