import pandas as pd

import geopandas as gpd



from shapely.geometry import LineString



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex2 import *
# Load the data and print the first 5 rows

birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])

print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))

birds_df.columns
# Your code here: Create the GeoDataFrame

birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df['location-long'], birds_df['location-lat']))



# Your code here: Set the CRS to {'init': 'epsg:4326'}

birds.crs = {"init": 'epsg: 4326'}



birds.tail()
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin(['North America', 'South America'])]

americas.head()

americas.crs
# Your code here

ax= americas.plot(color='none', edgecolor='gray')

birds.plot(ax=ax, markersize=10)

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
# Your code here

end_gdf = ____



end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()

end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)

end_gdf.crs = {'init' :'epsg:4326'}



# Check your answer

q_3.check()
ax= americas.plot(color="whitesmoke", edgecolor='black', figsize=(10,10))

path_gdf.plot(color='lightgreen', markersize=2, ax=ax)

start_gdf.plot(color='blue', markersize=10, ax=ax)

end_gdf.plot(color='red', ax=ax)
# Path of the shapefile to load

protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"



# Your code here

protected_areas = gpd.read_file(protected_filepath)





protected_areas.head()
# Country boundaries in South America

south_america = americas.loc[americas['continent']=='South America']



# Your code here: plot protected areas in South America

ax= south_america.plot(color='whitesmoke', edgecolor='black')

protected_areas.plot(ax=ax, color= 'green', markersize=10)

q_6.check()
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])

print("South America has {} square kilometers of protected areas.".format(P_Area))
south_america.head()
# Your code here: Calculate the total area of South America (in square kilometers)

totalArea =sum(south_america.geometry.to_crs(epsg=3035).area )/ 10**6





print("Area of South America: {} square kilometers".format(totalArea))





q_7.check()
# What percentage of South America is protected?

percentage_protected = P_Area/totalArea

print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')

protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1, color='lightgreen')

birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)


q_8.check()