# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium



#df_traffic = pd.read_csv('../input/ukTrafficAADF.csv')

df_acc = pd.read_csv('../input/permanent_send.csv', dtype=object)
heat_map = folium.Map(location=[51.5074, 0.1278],

                    zoom_start = 11) # Uses lat then lon. The bigger the zoom number, the closer in you get



from folium import plugins



# Adds tool to the top right

from folium.plugins import MeasureControl

heat_map.add_child(MeasureControl())



# Fairly obvious I imagine - works best with transparent backgrounds

from folium.plugins import FloatImage

url = ('https://media.licdn.com/mpr/mpr/shrinknp_100_100/AAEAAQAAAAAAAAlgAAAAJGE3OTA4YTdlLTkzZjUtNDFjYy1iZThlLWQ5OTNkYzlhNzM4OQ.jpg')

FloatImage(url, bottom=5, left=85).add_to(heat_map)



heat_map
from folium import plugins

from folium.plugins import HeatMap





heat_map = folium.Map(location=[51.5074, 0.1278],

                    zoom_start = 13) 



# handing it floats

df_acc['Latitude'] = df_acc['latitude_perm'].astype(float)

df_acc['Longitude'] = df_acc['longitude_perm'].astype(float)



# Filter the DF for rows, then columns, then remove NaNs

heat_df = df_acc 

heat_df = heat_df.dropna(axis=0, subset=['latitude_perm','longitude_perm'])



# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(heat_map)



# Display the map

heat_map


