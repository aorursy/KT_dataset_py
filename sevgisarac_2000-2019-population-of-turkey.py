#loading libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#this library necessary for map

from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json') as response:

    cities = json.load(response)

#cities["features"][0]
#load data sets

df = pd.read_csv('../input/trpopulation/TRNufus.csv')

df_im = pd.read_csv('../input/migratetr2/MigRate.csv')

#column name configuration, actually this is not necessary for this short visualization but it helps me for standardization

df.rename(columns = {'Number':'id','City':'city','Pop':'pop', 'Year':'year'},inplace =True)
#this step needed for json data and csv data merge, these two data sets merge with id

df.set_index('Id', inplace=True)
import plotly.express as px



fig = px.choropleth(df, geojson=cities, locations=df.index, color="pop", 

                    hover_name="city", animation_frame=df["year"],color_continuous_scale="earth",

                    

                         

                           labels={'pop':'population'}

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth_mapbox(df, geojson=cities, locations=df.index, color=np.log10(df["pop"]),hover_name="city", animation_frame=df["year"],

                           color_continuous_scale="Viridis",

                           

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": 38.963745, "lon": 35.243322},

                           opacity=0.7,

                           labels={'color':'population','Id': 'city','population':'pop'}

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth_mapbox(df, geojson=cities, locations=df.index, color=np.log10(df["pop"]),hover_name="city", animation_frame=df["year"],

                           color_continuous_scale='twilight',

                           

                           mapbox_style="carto-positron",

                           zoom=4, center = {"lat": 38.963745, "lon": 35.243322},

                           opacity=0.7,

                           labels={'color':'population','Id': 'city'}

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df_im.rename(columns = {'Id':'id','City':'city','Rate':'rate', 'Year':'year'},inplace =True)
df_im.set_index('id', inplace=True)
import plotly.express as px



fig = px.choropleth_mapbox(df_im, geojson=cities, locations=df_im.index, color='rate',hover_name="city", animation_frame=df_im["year"],

                           color_continuous_scale='twilight',

                           

                           mapbox_style="carto-positron",

                           zoom=5, center = {"lat": 38.963745, "lon": 35.243322},

                           opacity=0.7,

                           labels={'color':'rate','id': 'city'}

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
