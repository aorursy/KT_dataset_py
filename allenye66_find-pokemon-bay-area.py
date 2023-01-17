import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os
os.listdir("/kaggle/input/sf-bay-area-pokemon-go-spawns")
map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)
df = pd.read_csv('../input/sf-bay-area-pokemon-go-spawns/pokemon-spawns.csv')
df.head()
df.tail()
df.shape
df.head()
heat_data = [[row['lat'],row['lng']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(map_van)
map_van
#for index, row in df.iterrows():
#   print(df['lng'])
df.shape
df = df[df['lat'] > 36]  
df.shape
df = df[df['lng'] < -121]
df.shape
heat_data = [[row['lat'],row['lng']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(map_van)
map_van