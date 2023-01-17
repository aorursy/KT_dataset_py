import numpy as np

import pandas as pd

import geopandas as gpd

from keplergl import KeplerGl



import os

print(os.listdir("../input"))
df = gpd.read_file("../input/Information for Accommodation.csv")

# df = df.rename({'$Logitiute':'Longitude'}, axis='columns')



df['Longitude'] = pd.to_numeric(df['Logitiute'], errors='coerce')

df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')



gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)) ## To feed to keplerGl



gdf.head()
#Config values were tuned manually, then saved to a file.

config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': 'ts82sen', 'type': 'hexagon', 'config': {'dataId': 'Rooms', 'label': 'Point', 'color': [137, 218, 193], 'columns': {'lat': 'Latitude', 'lng': 'Longitude'}, 'isVisible': True, 'visConfig': {'opacity': 0.98, 'worldUnitSize': 0.25, 'resolution': 8, 'colorRange': {'name': 'Uber Viz Diverging 1.5', 'type': 'diverging', 'category': 'Uber', 'colors': ['#00939C', '#5DBABF', '#BAE1E2', '#F8C0AA', '#DD7755', '#C22E00'], 'reversed': False}, 'coverage': 1, 'sizeRange': [0, 500], 'percentile': [0, 100], 'elevationPercentile': [0, 100], 'elevationScale': 55.3, 'colorAggregation': 'sum', 'sizeAggregation': 'average', 'enable3d': True}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'Rooms', 'type': 'integer'}, 'colorScale': 'quantile', 'sizeField': {'name': 'Rooms', 'type': 'integer'}, 'sizeScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'Rooms': ['Name']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': []}, 'mapState': {'bearing': 21.035294117647066, 'dragRotate': True, 'latitude': 6.927406391215935, 'longitude': 80.63067683314334, 'pitch': 52.69321809155315, 'zoom': 7.529824405744526, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'mapStyles': {}}}}



map_pointChart = KeplerGl(data={'Rooms': gdf}, height=600, width=800)

map_pointChart.config = config

map_pointChart
# Save map_3DBuildings config to a file

# This utility function was used to save the map config, tuned via interactive map at development time.

with open('Tourist_Rooms_3d.py', 'w') as f:

   f.write('config = {}'.format(map_pointChart.config))