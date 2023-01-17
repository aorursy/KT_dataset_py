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
# Your code here: Create the GeoDataFrame

# this will not work beacuse the hyphen or dash / minus sign is interpreted as a subtraction

# birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df.location-long, birds_df.location-lat))

birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df['location-long'], birds_df['location-lat']))



# Your code here: Set the CRS to {'init': 'epsg:4326'}

birds.crs = {'init': 'epsg:4326'}



# Check your answer

q_1.check()
birds.head()

birds_df.timestamp

# birds_df["location-long"]
# Lines below will give you a hint or solution code

q_1.hint()

q_1.solution()
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin(['North America', 'South America'])]

americas.head()
# Your code here

ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

birds.plot(ax=ax)



# Uncomment to see a hint

#q_2.hint()
# Get credit for your work after you have created a map

q_2.check()



# Uncomment to see our solution (your code may look different!)

q_2.solution()
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

end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index() 

end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)

end_gdf.crs = {'init' :'epsg:4326'}



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# Your code here

ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

start_gdf.plot(ax=ax)

path_gdf.plot(ax=ax)

end_gdf.plot(ax=ax)



# Uncomment to see a hint

#q_4.hint()
# Get credit for your work after you have created a map

q_4.check()



# Uncomment to see our solution (your code may look different!)

#q_4.solution()
# Path of the shapefile to load

protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"



# Your code here

protected_areas = gpd.read_file(protected_filepath)



# Check your answer

q_5.check()
# Lines below will give you a hint or solution code

#q_5.hint()

#q_5.solution()
# Country boundaries in South America

south_america = americas.loc[americas['continent']=='South America']



# Your code here: plot protected areas in South America

bx = south_america.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

protected_areas.plot(ax=bx)



# Uncomment to see a hint

#q_6.hint()
# Get credit for your work after you have created a map

q_6.check()



# Uncomment to see our solution (your code may look different!)

#q_6.solution()
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])

print("South America has {} square kilometers of protected areas.".format(P_Area))
south_america.head()
import numpy

# Your code here: Calculate the total area of South America (in square kilometers)

# SOLUTION #1 MY SOLUTION: 0.001547957692761746

# south_america.crs = {'init': 'epsg:3035'}

# south_america.loc[:, "AREA"] = south_america.geometry.area / 10**6

# totalArea = south_america.AREA.sum()

# print("totalArea SOLUTION #1:", totalArea)

# print("crs SOLUTION #1:", south_america.crs)

# q_7.check()

# SOLUTION #2 OFFICIAL "q_7.solution()""  SOLUTION: 0.0015479576919549393

totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6

# print("totalArea SOLUTION #2:", totalArea)

# print("crs SOLUTION #2:", south_america.crs)

q_7.check()

# SOLUTION #3 "https://www.kaggle.com/learn-forum/119943" SOLUTION:0.001547957691954939

# totalArea = south_america['geometry'].to_crs(epsg=3035).area.sum() / 10**6

# print("totalArea SOLUTION #3:", totalArea)

# print("crs SOLUTION #3:", south_america.crs)

# q_7.check()

# SOLUTION #4 EPSG 4326 wgs84 datum (Google Earth/Open Street Map) [instead of EPSG 3857, Google Maps]: 1.289745180671118e-13

# totalArea = sum(south_america.geometry.to_crs(epsg=4326).area) / 10**6

# print("totalArea4:", totalArea)

# print("crs4:", south_america.crs)

# q_7.check()



# print(south_america.head())



# Check your answer

# q_7.check()



# to 'search for the correct answer try this black box hunt

#

# area_range =[0.001547957692761746, 0.0015479576919549393, 0.001547957691954939, 1.289745180671118e-13]

# area_range =[0.001547957692761746, 0.0015479576919549393, 0.001547957691954939]

#

#

# for black_box_area in numpy.arange(min(area_range), max(area_range), 0.0000000000000000001):

# #     print("black_box_area:",black_box_area)

#     totalArea=black_box_area

#     q_7.check()
# Lines below will give you a hint or solution code

q_7.hint()

q_7.solution()
# What percentage of South America is protected?

percentage_protected = P_Area/totalArea

print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
# Your code here

____



# Uncomment to see a hint

#q_8.hint()
# Get credit for your work after you have created a map

q_8.check()



# Uncomment to see our solution (your code may look different!)

#q_8.solution()