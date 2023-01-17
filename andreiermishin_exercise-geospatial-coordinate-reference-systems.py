import pandas as pd

import geopandas as gpd



from shapely.geometry import LineString



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex2 import *
# Load the data and print the first 5 rows

birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])

print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))

birds_df.head()
# Convert the DataFrame to a GeoDataFrame

birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df['location-long'], birds_df['location-lat']))



# Set the coordinate reference system (CRS) to EPSG 4326

birds.crs = {'init': 'epsg:4326'}



# Check your answer

q_1.check()
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows:

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin(['North America', 'South America'])]

americas.head()
ax = americas.plot(figsize=(8,8), color='white', linestyle=':', edgecolor='gray')

birds.plot(ax=ax, markersize=10)



# Uncomment to zoom in

ax.set_xlim([-110, -30])

ax.set_ylim([-30, 60])
# Get credit for your work after you have created a map

q_2.check()
# GeoDataFrame showing path for each bird

path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()

path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)

path_gdf.crs = {'init' :'epsg:4326'}



# GeoDataFrame showing starting point for each bird

start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()

start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)

start_gdf.crs = {'init' :'epsg:4326'}



# Show first five rows of GeoDataFrame

start_gdf.head()
# GeoDataFrame showing end point for each bird

end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()

end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)

end_gdf.crs = {'init' :'epsg:4326'}



# Check your answer

q_3.check()

end_gdf.head()
ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

start_gdf.plot(ax=ax, color='green',  markersize=30)

path_gdf.plot(ax=ax, cmap='tab20b', linestyle='-', linewidth=1, zorder=1)

end_gdf.plot(ax=ax, markersize=30)



# Uncomment to zoom in

ax.set_xlim([-110, -30])

ax.set_ylim([-30, 60])
# Get credit for your work after you have created a map

q_4.check()
# Path of the shapefile to load

protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"

protected_areas = gpd.read_file(protected_filepath)



# Check your answer

q_5.check()
# Country boundaries in South America

south_america = americas.loc[americas['continent']=='South America']



ax = south_america.plot(figsize=(8,8), color='white', edgecolor='gray')

protected_areas.plot(ax=ax, color='green', alpha=0.5)
# Get credit for your work after you have created a map

q_6.check()
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])

print("South America has {} square kilometers of protected areas.".format(P_Area))
south_america.head()
# Your code here: Calculate the total area of South America (in square kilometers)

# totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6

totalArea = south_america['geometry'].to_crs(epsg=3035).area.sum() / 10**6



# Check your answer

q_7.check()
# What percentage of South America is protected?

percentage_protected = P_Area / totalArea

print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
ax = south_america.plot(figsize=(10, 10), color='white', edgecolor='gray')

protected_areas[ protected_areas['MARINE']!='2' ].plot(ax=ax, color='green', alpha=0.5, zorder=1)

birds[ birds['geometry'].y < 0 ].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)



# Uncomment to zoom in

ax.set_ylim([-30, 10])
# Get credit for your work after you have created a map

q_8.check()