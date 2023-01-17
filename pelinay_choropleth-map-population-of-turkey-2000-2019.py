import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import matplotlib.pyplot as plt

tur_map = pd.read_csv("../input/tur-map2/TRNufus.csv",sep=",")

# check data type so we can see that this is not a normal dataframe, but a GEOdataframe

tur_map.head()
from urllib.request import urlopen

import json

with open('../input/trcities/tr-cities-utf8.json' ) as f:

    cities = json.load(f)
#Describing features

feat_desc = pd.DataFrame({'Description': tur_map.columns, 

                          'Values': [tur_map[i].unique() for i in tur_map.columns],

                          'Number of unique values': [len(tur_map[i].unique()) for i in tur_map.columns]})

feat_desc
#missing data

total = tur_map.isnull().sum().sort_values(ascending=False)

percent = (tur_map.isnull().sum()/tur_map.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
tur_map.rename(columns = {"Id": "id", 

                     "City":"city","Pop":"pop","Year":"year"}, 

                                 inplace = True) 
scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'], \

       [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds


import plotly.express as px



fig = px.choropleth_mapbox(tur_map, geojson=cities, locations=tur_map[['id']], color = tur_map["pop"],

                           hover_name="city", 

                           animation_frame=tur_map["year"],

                           color_continuous_scale=scl,                           

                           mapbox_style="carto-positron",

                           zoom=4.5, center = {"lat": 38.963745, "lon": 35.243322},

                           opacity=0.5,

                           labels={'pop':'population'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()