# OK, let me call my friends

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir('../input/'))
data = pd.read_csv('../input/open_pubs.csv')
# Check the feeling

data.head()
data.drop(data.index[data.latitude == '\\N'], inplace = True)

data.drop(data.index[data.longitude == '\\N'], inplace = True)

data.drop(data.index[data.longitude == 'Broxbourne'], inplace = True)

data.drop(data.index[data.longitude == 'Ryedale'], inplace = True)
data.info()
data['latitude_float'] = data['latitude'].astype(np.float64)

data['longitude_float'] = data['longitude'].astype(np.float64)
data.head(5)
import folium

map = folium.Map(

    location=[52.204990, 0.122139]

    , tiles = 'OpenStreetMap'

    , zoom_start = 9

)



for each in data[:1000].iterrows():

    folium.CircleMarker([each[1]['latitude_float'],

                         each[1]['longitude_float']],

                        radius = 5,

                        color = 'blue',

                        popup = str(each[1]['name']) + '\\n' + str(each[1]['address']),

                        fill_color = '#FD8A6C'

                       ).add_to(map)



map