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

def my_geocoder(row):

    point = geocode(row, provider='nominatim').geometry[0]

    return pd.Series({'Longitude': point.x, 'Latitude': point.y})



berkeley_locations = rows_with_missing.apply(lambda x: my_geocoder(x['Address']), axis=1)

starbucks.update(berkeley_locations)

# Check your answer

q_1.check()
# Create a base map

m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)



# Your code here: Add a marker for each Berkeley location



for idx, row in starbucks[starbucks["City"]=='Berkeley'].iterrows():

    Marker([row['Latitude'], row['Longitude']]).add_to(m_2)

# Uncomment to see a hint

#q_2.a.hint()

m_2

# Show the map

#embed_map(m_2, 'q_2.html')
CA_counties = gpd.read_file("../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp")

CA_counties.head()
CA_pop = pd.read_csv("../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID")

CA_high_earners = pd.read_csv("../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID")

CA_median_age = pd.read_csv("../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID")

CA_pop.head()

CA_high_earners.head()

CA_median_age.head()

CA_counties.head()




cols_to_add = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()

CA_stats = CA_counties.merge(cols_to_add, on="GEOID")
# Lines below will give you a hint or solution code

#q_3.hint()

q_3.check()


CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]



CA_stats.head()
# Your code here

sel_counties =CA_stats.loc[(CA_stats.density >285 ) & (CA_stats.median_age < 38.5)& (CA_stats.high_earners > 100000) & 

                          ((CA_stats.density >1400 ) | (CA_stats.median_age < 35.5)| (CA_stats.high_earners > 500000))]

sel_counties.head()

# Check your answer

q_4.check()
starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks.Longitude, starbucks.Latitude))

starbucks_gdf.crs = {'init': 'epsg:4326'}

starbucks_gdf.head()
# Fill in your answer

location_of_interst= gpd.sjoin(starbucks_gdf,sel_counties)

num_stores = len(location_of_interst)

print(num_stores)

# Check your answer

q_5.check()
# Create a base map

m_6 = folium.Map(location=[37,-120], zoom_start=7)



# Your code here: show selected store locations





mc = MarkerCluster()

for idx, row in location_of_interst.iterrows():

    

    mc.add_child(Marker([row['Latitude'], row['Longitude']]))

m_6.add_child(mc)



# Uncomment to see a hint

#q_6.hint()



m_6