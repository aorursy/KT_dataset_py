# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import math
import random
import datetime
import time

import requests
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar

import folium
from folium import plugins
# ingest data
df_mx = pd.read_csv('../input/demographic-data-mexico-submission-1/1_min.csv')

# read
with open('../input/demographic-data-mexico-submission-1/municipality_geo-json_shapes.json') as f:
    geo_json_mx = json.load(f)
# add a total population column
df_mx['Total Population: 2010'] = df_mx['Total Male Population: 2010'] + df_mx['Total Female Population: 2010']


df_mx['Code'].astype(str)

df_mx['Code'] = df_mx.apply( lambda row: str(row['Code']) if len(str(row['Code']))==5 else '0'+str(row['Code']), axis=1 )

df_mx['State Code'].astype(str)

df_mx['State Code'] = df_mx.apply( lambda row: str(row['State Code']) if len(str(row['State Code']))==2 else '0'+str(row['State Code']), axis=1 )
latCen_mx = +023.00
lonCen_mx = -100.00

indicaux = 'Percentage of Females of 15 to 29: 2015' # chose the indicator to plot

# define a map
map_mx = folium.Map(location=[latCen_mx, lonCen_mx], width=970, height=600, zoom_start=5, min_zoom=5, max_zoom=10)
# add colors to the map

map_mx.choropleth(geo_data=geo_json_mx, data=df_mx,
    columns=['Code', indicaux],
    key_on='feature.properties.mun_code',
    fill_color='YlGnBu', fill_opacity=0.7, 
    line_opacity=0.2
)
# display map
map_mx