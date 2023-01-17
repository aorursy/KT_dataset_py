from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex2 import *

print("Setup is completed!")



import pandas as pd

import geopandas as gpd

from shapely.geometry import LineString
# load data

birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])

print(f"There are {birds_df['tag-local-identifier'].nunique()} different birds in the dataset.")

birds_df.head()
# create the GeoDataFrame

birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df['location-long'], birds_df['location-lat']))



# set the CRS to {'init': 'epsg:4326'}

birds.crs = {'init': 'epsg:4326'}



# Check your answer

q_1.check()
# lines below will give you a hint or solution code

# q_1.hint()

# q_1.solution()
# load a GeoDataFrame with country boundaries in North/South America

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin(['North America', 'South America'])]

americas.head()
ax = americas.plot(figsize=(10,10), color="whitesmoke", linestyle=":", edgecolor="lightgray")

birds.plot(ax=ax, color="black", markersize=2)



# uncomment to see a hint

# q_2.hint()
# get credit for your work after you have created a map

q_2.check()



# uncomment to see our solution (your code may look different!)

# q_2.solution()
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
end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()

end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)

end_gdf.crs = {'init' :'epsg:4326'}



# check your answer

q_3.check()
# lines below will give you a hint or solution code

# q_3.hint()

# q_3.solution()
# Your code here

ax = americas.plot(figsize=(10,10), color="whitesmoke", linestyle=":", edgecolor="lightgray")



start_gdf.plot(ax=ax, color='red', markersize=20)

path_gdf.plot(ax=ax, cmap='tab20b', linestyle='-', linewidth=1, zorder=1)

end_gdf.plot(ax=ax, color='black', markersize=20)



# uncomment to see a hint

# q_4.hint()
# get credit for your work after you have created a map

q_4.check()



# uncomment to see our solution (your code may look different!)

# q_4.solution()
# load the protected_areas shapefile

protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"

protected_areas = gpd.read_file(protected_filepath)



# check your answer

q_5.check()
# lines below will give you a hint or solution code

# q_5.hint()

# q_5.solution()
# Country boundaries in South America

south_america = americas.loc[americas['continent']=='South America']



# plot protected areas in South America

ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')

protected_areas.plot(ax=ax, alpha=0.4)



# Uncomment to see a hint

# q_6.hint()
# get credit for your work after you have created a map

q_6.check()



# uncomment to see our solution (your code may look different!)

# q_6.solution()
P_Area = sum(protected_areas['REP_AREA'] - protected_areas['REP_M_AREA'])

print(f"South America has {P_Area} square kilometers of protected areas.")
south_america.head()
# calculate the total area of South America (in square kilometers)

totalArea = sum(south_america['geometry'].to_crs(epsg=3035).area) / 10**6



# check your answer

q_7.check()
# lines below will give you a hint or solution code

# q_7.hint()

# q_7.solution()
# What percentage of South America is protected?

percentage_protected = P_Area/totalArea

print(f'Approximately {round(percentage_protected*100, 2)}% of South America is protected.')
birds.head()
ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')



protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)

birds[birds['geometry'].y < 0].plot(ax=ax, color='black', alpha=0.6, markersize=10, zorder=2)



# uncomment to see a hint

# q_8.hint()
# get credit for your work after you have created a map

q_8.check()



# uncomment to see our solution (your code may look different!)

# q_8.solution()