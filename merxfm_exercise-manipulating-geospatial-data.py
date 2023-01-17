import math

import pandas as pd

import geopandas as gpd

#from geopandas.tools import geocode            # What you'd normally run

from learntools.geospatial.tools import geocode # Just for this exercise



import folium 

from folium import Marker

from folium.plugins import MarkerCluster



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex4 import *
# Load and preview Starbucks locations in California

starbucks = pd.read_csv("../input/geospatial-learn-course-data/starbucks_locations.csv")

starbucks.head()
# How many rows in each column have missing values?

print(starbucks.isnull().sum())



# View rows with missing locations

rows_with_missing = starbucks[starbucks["City"]=="Berkeley"]

rows_with_missing
# Your code here

for index, row in rows_with_missing.iterrows():

    starbucks.loc[index, 'Longitude'] = geocode(row['Address'], provider='nominatim').geometry.iloc[0].x

    starbucks.loc[index, 'Latitude'] = geocode(row['Address'], provider='nominatim').geometry.iloc[0].y

    



# Check your answer

q_1.check()
# Line below will give you solution code

q_1.solution()
# Create a base map

m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)



berkeley = starbucks[starbucks.City == 'Berkeley']



# Your code here: Add a marker for each Berkeley location

for index, row in berkeley.iterrows():

    Marker([row['Latitude'], row['Longitude']]).add_to(m_2)



# Uncomment to see a hint

#q_2.a.hint()



# Show the map

m_2
# Get credit for your work after you have created a map

q_2.a.check()
# View the solution (Run this code cell to receive credit!)

q_2.b.solution()
CA_counties = gpd.read_file("../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp")

CA_counties.head()
CA_pop = pd.read_csv("../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID")

CA_high_earners = pd.read_csv("../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID")

CA_median_age = pd.read_csv("../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID")
# Your code here

CA_three = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()

CA_stats = CA_counties.merge(CA_three, on='GEOID')



# Check your answer

q_3.check()
CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]
# Your code here



and_mask = (CA_stats.high_earners >= 1e5) & (CA_stats.median_age < 38.5) & (CA_stats.density >= 285)

or_mask = (CA_stats.high_earners >= 5e5) | (CA_stats.median_age < 35.5) | (CA_stats.density >= 1400)



sel_counties = CA_stats[(and_mask) & (or_mask)]



# Check your answer

q_4.check()
starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks.Longitude, starbucks.Latitude))

starbucks_gdf.crs = {'init': 'epsg:4326'}
# Fill in your answer

selected_stores = gpd.sjoin(starbucks_gdf, sel_counties)



display(selected_stores.head())



num_stores = len(selected_stores)



# Check your answer

q_5.check()
# Create a base map

m_6 = folium.Map(location=[37,-120], zoom_start=6)



# Your code here: show selected store locations

for index, row in selected_stores.iterrows():

    Marker([row['Latitude'], row['Longitude']]).add_to(m_6)



# Uncomment to see a hint

#q_6.hint()



# Show the map

m_6
# Get credit for your work after you have created a map

q_6.check()



# Uncomment to see our solution (your code may look different!)

#q_6.solution()