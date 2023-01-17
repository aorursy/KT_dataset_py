import pandas as pd

import geopandas as gpd



from shapely.geometry import LineString



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex2 import *
# Load the data and print the first 5 rows

# reading a csv with longitude and latitude coordinates into a dataframe as well as parsing the timestamp column to date format

birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])

# tag-local-identifier = bird id, here we are looking for the number of unique identifiers = # of birds

print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))

birds_df.head()
# Your code here: Create the GeoDataFrame

# make a GeoDataFrame from a regular dataframe, the geometry column contains POINT values of the longitude and latitude

birds = gpd.GeoDataFrame(

    birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))



# Your code here: Set the CRS to {'init': 'epsg:4326'}

# set the correct projection

birds.crs = {'init': 'epsg:4326'}



# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# only looking for the continents of North and Sourht America

americas = world.loc[world['continent'].isin(['North America', 'South America'])]

americas.head()
# Your code here

# first plot the countries and then the bird locations

ax = americas.plot(figsize=(8,8), color='whitesmoke', linestyle=':', edgecolor='black')

birds.plot(markersize=1, ax=ax)



# Uncomment to see a hint

#q_2.hint()
# Get credit for your work after you have created a map

q_2.check()



# Uncomment to see our solution (your code may look different!)

#q_2.solution()
# GeoDataFrame showing path for each bird

# group by tag-local-identifier then convert the geometry column to a list (of points) for each tag-id

# with the list of points create a LineString object in the geometry column and reset the index

path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()

# create a geodataframe

path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)

path_gdf.crs = {'init' :'epsg:4326'}



# GeoDataFrame showing starting point for each bird = the starting point is the first one in the list by tag-id (x[0])

start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()

start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)

start_gdf.crs = {'init' :'epsg:4326'}



# Show first five rows of GeoDataFrame

start_gdf.head()
# Your code here

# GeoDataFrame showing starting point for each bird = the end point is the last one in the list by tag-id (x[-1])

end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()

end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)

end_gdf.crs = {'init' :'epsg:4326'}



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# Your code here

# first plot the countries and then the bird starting and end points as well as the path

ax = americas.plot(figsize=(14,12), color='whitesmoke', linestyle=':', edgecolor='black')

start_gdf.plot(color='red',markersize=5, ax=ax)

end_gdf.plot(color='blue',markersize=5, ax=ax)

path_gdf.plot(color='green',markersize=1, ax=ax)



# Uncomment to see a hint

#q_4.hint()
# Get credit for your work after you have created a map

q_4.check()



# Uncomment to see our solution (your code may look different!)

#q_4.solution()
# Path of the shapefile to load

protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"



# Your code here

# convert shape file to geodataframe

protected_areas = gpd.read_file(protected_filepath)



# Check your answer

q_5.check()
# Lines below will give you a hint or solution code

#q_5.hint()

#q_5.solution()
# Country boundaries in South America

south_america = americas.loc[americas['continent']=='South America']



# Your code here: plot protected areas in South America

# Create a map

ax = south_america.plot(figsize=(12,12), color='whitesmoke', linestyle=':', edgecolor='black')

protected_areas.plot(alpha=0.4, ax=ax)



# Uncomment to see a hint

#q_6.hint()
# Get credit for your work after you have created a map

q_6.check()



# Uncomment to see our solution (your code may look different!)

#q_6.solution()
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])

print("South America has {} square kilometers of protected areas.".format(P_Area))
south_america.head()
# Your code here: Calculate the total area of South America (in square kilometers), we need to convert square meters to square km's

totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6 



# Check your answer

q_7.check()
# Lines below will give you a hint or solution code

#q_7.hint()

#q_7.solution()
# What percentage of South America is protected?

percentage_protected = P_Area/totalArea

print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
# Your code here

# plot the south american country boundaries

ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')

# plot the protected areas

protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)

# plot all birds with coordinates in the southern hemisphere => geometry.y < 0

birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)



# Uncomment to see a hint

#q_8.hint()
# Get credit for your work after you have created a map

q_8.check()



# Uncomment to see our solution (your code may look different!)

#q_8.solution()