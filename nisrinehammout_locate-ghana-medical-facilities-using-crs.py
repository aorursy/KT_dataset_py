import geopandas as gpd

import pandas as pd
# Load a GeoDataFrame containing regions in Ghana

regions = gpd.read_file("../input/geospatial-learn-course-data/ghana/ghana/Regions/Map_of_Regions_in_Ghana.shp")

print(regions.crs)

regions.head()
# Create a DataFrame with health facilities in Ghana

facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")

facilities_df.head()
facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))

# Setting the CRS cordinate reference system to EPSG 4326 



facilities.crs={"init": 'epsg: 4326'}

#changing the dataset from pandas to geopandas

print(type(facilities), type(facilities_df))

facilities.head()

ax = regions.to_crs(epsg=4326).plot(figsize=(8,8), color='whitesmoke', linestyle=':', edgecolor='black')

facilities.plot(markersize=1, ax=ax)
facilities.geometry.x.head()
# Calculate the area (in square meters) of each polygon in the GeoDataFrame 

regions.loc[:, "AREA"] = regions.geometry.area / 10**6



print("Area of Ghana: {} square kilometers".format(regions.AREA.sum()))

print("CRS:", regions.crs)

regions.head()