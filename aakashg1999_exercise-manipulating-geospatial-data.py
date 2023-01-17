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
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
# Load and preview Starbucks locations in California

starbucks = pd.read_csv("../input/geospatial-learn-course-data/starbucks_locations.csv")

starbucks.head()
# How many rows in each column have missing values?

print(starbucks.isnull().sum())



# View rows with missing locations

rows_with_missing = starbucks[starbucks["City"]=="Berkeley"]

rows_with_missing
# Your code here

def my_geocode(row):

    result=geocode(row,provider='nominatim').iloc[0]

    latitude=result.geometry.y

    longitude=result.geometry.x

    new_df=pd.Series({"Longitude":longitude,"Latitude":latitude})

    return new_df

#print(result.geometry.x)



rows_with_missing[["Longitude","Latitude"]]=rows_with_missing.apply(lambda x: my_geocode(x["Address"]),axis=1)

starbucks.update(rows_with_missing)

# Check your answer

starbucks[150:157][:]

q_1.check()
# Line below will give you solution code

#q_1.solution()
# Create a base map

m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)



# Your code here: Add a marker for each Berkeley location

for idx,row in rows_with_missing.iterrows():

    Marker([row['Latitude'],row['Longitude']],popup=row['Store Number']).add_to(m_2)



# Uncomment to see a hint

#q_2.a.hint()



# Show the map

embed_map(m_2, 'q_2.html')
# Get credit for your work after you have created a map

q_2.a.check()



# Uncomment to see our solution (your code may look different!)

#q_2.a.solution()
# View the solution

q_2.b.solution()
CA_counties = gpd.read_file("../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp")

CA_counties.head()
CA_pop = pd.read_csv("../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID")

CA_high_earners = pd.read_csv("../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID")

CA_median_age = pd.read_csv("../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID")
# Your code here

CA_pop=CA_pop.merge(CA_median_age,on="GEOID")

CA_pop=CA_pop.merge(CA_high_earners,on="GEOID")

CA_stats=CA_counties.merge(CA_pop,on="GEOID")

CA_stats.head()

#CA_stats =



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]

CA_stats.crs = {'init': 'epsg:4326'}

CA_stats.head()
# Your code here

sel_counties = CA_stats[(CA_stats.median_age<38.5) & (CA_stats.density>=285) & (CA_stats.high_earners>=100000) ]

sel_counties=sel_counties[0:3]

# Check your answer

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()

sel_counties.head()
starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks.Longitude, starbucks.Latitude))

starbucks_gdf.crs = {'init': 'epsg:4326'}
# Fill in your answer

num_stores = len(gpd.sjoin(starbucks_gdf,sel_counties))

# Check your answer

q_5.check()
# Lines below will give you a hint or solution code

#q_5.hint()

#q_5.solution()
# Create a base map

m_6 = folium.Map(location=[37,-120], zoom_start=6)

new_li=gpd.sjoin(starbucks_gdf,sel_counties)

# Your code here: show selected store locations

for idx,row in new_li.iterrows():

    Marker([row['Latitude'],row['Longitude']],popup=row["Store Number"]).add_to(m_6)



# Uncomment to see a hint

#q_6.hint()



# Show the map

embed_map(m_6, 'q_6.html')
# Get credit for your work after you have created a map

q_6.check()



# Uncomment to see our solution (your code may look different!)

#q_6.solution()