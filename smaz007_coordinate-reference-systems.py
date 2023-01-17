

import geopandas as gpd

import pandas as pd
# Load a GeoDataFrame containing regions in Ghana

regions = gpd.read_file("../input/geospatial-learn-course-data/ghana/ghana/Regions/Map_of_Regions_in_Ghana.shp")

print(regions.crs)
# Create a DataFrame with health facilities in Ghana

facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")



# Convert the DataFrame to a GeoDataFrame

facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))



# Set the coordinate reference system (CRS) to EPSG 4326

facilities.crs = {'init': 'epsg:4326'}



# View the first five rows of the GeoDataFrame

facilities.head()
# Create a map

ax = regions.plot(figsize=(8,8), color='whitesmoke', linestyle='-', edgecolor='blue')

facilities.to_crs(epsg=32630).plot(markersize=1, ax=ax)
# The "Latitude" and "Longitude" columns are unchanged

facilities.to_crs(epsg=32630).head()
# Change the CRS to EPSG 4326

regions.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs").head()
# Get the x-coordinate of each point

facilities.geometry.x.head()
# Calculate the area (in square meters) of each polygon in the GeoDataFrame 

regions.loc[:, "AREA"] = regions.geometry.area / 10**6



print("Area of Ghana: {} square kilometers".format(regions.AREA.sum()))

print("CRS:", regions.crs)

regions.head()