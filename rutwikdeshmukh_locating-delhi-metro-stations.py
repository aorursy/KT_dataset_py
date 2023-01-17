import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/delhi-metro-stations-data/DELHI_METRO_DATA.csv')
df.head()
df.isnull().sum()
df = df.dropna()
df['Line'].value_counts()
latitude, longitude = df['Latitude'].median(), df['Longitude'].median()
import folium
map1 = folium.Map(location=[latitude, longitude], zoom_start=11)

for lat, long, label in zip(df['Latitude'], df['Longitude'], df['Station']):
    folium.CircleMarker(
        [lat,long],
        radius = 5,
        color = 'blue',
        popup = label,
        fill=True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(map1)
map1
