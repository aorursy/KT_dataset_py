import numpy as np

import pandas as pd

import geopandas as gpd

from keplergl import KeplerGl



import os

print(os.listdir("../input/nypd-complaint-data-current-ytd-july-2018"))

print(os.listdir("../input/nyc-neighborhoods-data"))

print(os.listdir("../input/san-fransisco-open-data-for-building-footprints"))
df = gpd.read_file("../input/nypd-complaint-data-current-ytd-july-2018/NYPD_Complaint_Data_Current_YTD.csv")



df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')



gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)) ## To feed to keplerGl



gdf.head()
config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': '0xm03v7', 'type': 'point', 'config': {'dataId': 'crimes', 'label': 'Point', 'color': [18, 147, 154], 'columns': {'lat': 'Latitude', 'lng': 'Longitude', 'altitude': None}, 'isVisible': True, 'visConfig': {'radius': 14.5, 'fixedRadius': False, 'opacity': 0.19, 'outline': False, 'thickness': 2, 'strokeColor': None, 'colorRange': {'name': 'ColorBrewer Set1-6', 'type': 'qualitative', 'category': 'ColorBrewer', 'colors': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'], 'reversed': False}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radiusRange': [0, 50], 'filled': True}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'SUSP_SEX', 'type': 'string'}, 'colorScale': 'ordinal', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear'}}, {'id': '280ug18', 'type': 'geojson', 'config': {'dataId': 'crimes', 'label': 'crimes', 'color': [221, 178, 124], 'columns': {'geojson': 'Lat_Lon'}, 'isVisible': True, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None, 'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True, 'filled': False, 'enable3d': False, 'wireframe': False}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}, {'id': 'y4fjza', 'type': 'geojson', 'config': {'dataId': 'crimes', 'label': 'crimes', 'color': [136, 87, 44], 'columns': {'geojson': 'geometry'}, 'isVisible': True, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None, 'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'stroked': False, 'filled': True, 'enable3d': False, 'wireframe': False}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'crimes': ['CMPLNT_NUM', 'ADDR_PCT_CD', 'BORO_NM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': []}, 'mapState': {'bearing': 0, 'dragRotate': False, 'latitude': 40.710948277451024, 'longitude': -73.95360501205884, 'pitch': 0, 'zoom': 13.19223566766781, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'mapStyles': {}}}}



map_pointChart = KeplerGl(data={'crimes': gdf}, height=600, width=800)

map_pointChart.config = config

map_pointChart
config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': 'bxpdchi', 'type': 'hexagon', 'config': {'dataId': 'crimes', 'label': 'crimes', 'color': [183, 136, 94], 'columns': {'lat': 'Latitude', 'lng': 'Longitude'}, 'isVisible': True, 'visConfig': {'opacity': 0.8, 'worldUnitSize': 1, 'resolution': 8, 'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#FFC300', '#F1920E', '#E3611C', '#C70039', '#900C3F', '#5A1846'], 'reversed': True}, 'coverage': 1, 'sizeRange': [0, 500], 'percentile': [0, 100], 'elevationPercentile': [0, 100], 'elevationScale': 5, 'colorAggregation': 'count', 'sizeAggregation': 'count', 'enable3d': False}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'crimes': ['CMPLNT_NUM', 'ADDR_PCT_CD', 'BORO_NM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': []}, 'mapState': {'bearing': 0, 'dragRotate': False, 'latitude': 40.69948531525293, 'longitude': -74.1720210730601, 'pitch': 0, 'zoom': 9.355823249249143, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'mapStyles': {}}}}



map_Hexogon = KeplerGl(data={'crimes': gdf}, height=600, width=800)

map_Hexogon.config = config

map_Hexogon
config = {'version': 'v1', 'config': {'visState': {'filters': [{'dataId': 'crimes', 'id': 'mprnycc1o', 'name': 'CMPLNT_FR_TM', 'type': 'timeRange', 'value': [1585483170000, 1585485990000], 'enlarged': True, 'plotType': 'histogram', 'yAxis': None}], 'layers': [{'id': 't6rsxg', 'type': 'point', 'config': {'dataId': 'crimes', 'label': 'Point', 'color': [255, 254, 230], 'columns': {'lat': 'Latitude', 'lng': 'Longitude', 'altitude': None}, 'isVisible': True, 'visConfig': {'radius': 0.3, 'fixedRadius': False, 'opacity': 0.8, 'outline': False, 'thickness': 2, 'strokeColor': None, 'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radiusRange': [0, 50], 'filled': True}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear'}}, {'id': 'b6l82wo', 'type': 'point', 'config': {'dataId': 'crimes', 'label': 'crimes', 'color': [255, 254, 230], 'columns': {'lat': 'Latitude', 'lng': 'Longitude', 'altitude': None}, 'isVisible': True, 'visConfig': {'radius': 3.8, 'fixedRadius': False, 'opacity': 0.8, 'outline': False, 'thickness': 2, 'strokeColor': None, 'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radiusRange': [0, 50], 'filled': True}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'crimes': ['CMPLNT_NUM', 'ADDR_PCT_CD', 'BORO_NM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': []}, 'mapState': {'bearing': 0, 'dragRotate': False, 'latitude': 40.71327133151427, 'longitude': -74.1913923525195, 'pitch': 0, 'zoom': 8.715569801244058, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'mapStyles': {}}}}



map_TimeSeriesFiltering = KeplerGl(data={'crimes': gdf}, height=600, width=800)

map_TimeSeriesFiltering.config = config

map_TimeSeriesFiltering
def count_incidents_neighborhood(data, neighborhoods):

    # spatial join and group by to get count of incidents in each poneighbourhood 

    joined = gpd.sjoin(gdf, neighborhoods, op='within')

    grouped = joined.groupby('neighborhood').size()

    df = grouped.to_frame().reset_index()

    df.columns = ['neighborhood', 'count']

    merged = neighborhoods.merge(df, on='neighborhood', how='outer')

    merged['count'].fillna(0,inplace=True)

    merged['count'] = merged['count'].astype(int)

    return merged
# 3 - Neighbourhoods

geojson_file = '../input/nyc-neighborhoods-data/newyork_neighborhoods.geojson'

neighborhoods = gpd.read_file(geojson_file)



neighborhoods.head()
!conda install -y -c conda-forge/label/cf202003 rtree
merged = count_incidents_neighborhood(gdf, neighborhoods)

merged.head()
config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': '3qw013f', 'type': 'geojson', 'config': {'dataId': 'NeighborhoodCrimes', 'label': 'NeighborhoodCrimes', 'color': [18, 147, 154], 'columns': {'geojson': 'geometry'}, 'isVisible': True, 'visConfig': {'opacity': 0.5, 'thickness': 0.5, 'strokeColor': [221, 178, 124], 'colorRange': {'name': 'ColorBrewer PuBu-6', 'type': 'sequential', 'category': 'ColorBrewer', 'colors': ['#f1eef6', '#d0d1e6', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d'], 'reversed': False}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'count', 'type': 'integer'}, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'NeighborhoodCrimes': ['neighborhood', 'boroughCode', 'borough', '@id', 'count']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': []}, 'mapState': {'bearing': 0, 'dragRotate': False, 'latitude': 40.711662196251034, 'longitude': -74.02936469973493, 'pitch': 0, 'zoom': 9.451941083083048, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'mapStyles': {}}}}



map_NeighnorHoodCrimes = KeplerGl(data={'NeighborhoodCrimes': merged}, height=600, width=800)

map_NeighnorHoodCrimes.config = config

map_NeighnorHoodCrimes
df = gpd.read_file("../input/san-fransisco-open-data-for-building-footprints/Building_Footprints.csv")

df.head()
from shapely import wkt



df['shape'] = df['shape'].apply(wkt.loads)
# df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')

# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)) ## To feed to keplerGl



buildings = gpd.GeoDataFrame(df, geometry='shape')

buildings.head()
config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': 'noq9dns', 'type': 'geojson', 'config': {'dataId': 'Buildings', 'label': 'Buildings', 'color': [18, 147, 154], 'columns': {'geojson': 'shape'}, 'isVisible': True, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': [221, 178, 124], 'colorRange': {'name': 'Ice And Fire', 'type': 'diverging', 'category': 'Uber', 'colors': ['#0198BD', '#49E3CE', '#E8FEB5', '#FEEDB1', '#FEAD54', '#D50255'], 'reversed': False}, 'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 3, 'stroked': False, 'filled': True, 'enable3d': True, 'wireframe': False}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': {'name': 'hgt_median_m', 'type': 'real'}, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'strokeColorField': None, 'strokeColorScale': 'quantile', 'heightField': {'name': 'hgt_median_m', 'type': 'real'}, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'Buildings': ['sf16_bldgid', 'area_id', 'mblr', 'p2010_name', 'p2010_zminn88ft']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': []}, 'mapState': {'bearing': 24, 'dragRotate': True, 'latitude': 37.78074167677063, 'longitude': -122.43457979576556, 'pitch': 50, 'zoom': 12.15693360108208, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'mapStyles': {}}}}



map_3DBuildings= KeplerGl(data={'Buildings': buildings[:10000]}, height=600, width=800)

map_3DBuildings.config = config

map_3DBuildings
# Save map_3DBuildings config to a file

# This utility function was used to save the map config, tuned via interactive map at development time.

with open('map_3DBuildings.py', 'w') as f:

   f.write('config = {}'.format(map_3DBuildings.config))