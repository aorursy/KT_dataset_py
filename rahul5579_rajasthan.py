#Package Imports
import folium
import numpy as np
import pandas as pd
#Mapping Location of Rajasthan
map_osm = folium.Map(location=[26.5576148,69.3786444],zoom_start=6)
map_osm
df = pd.read_csv('census.csv')
df.shape