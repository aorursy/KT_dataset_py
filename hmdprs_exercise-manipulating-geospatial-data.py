from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex4 import *

print('Setup is completed')



import math

import pandas as pd

import geopandas as gpd

# from geopandas.tools import geocode           # What you'd normally run

from learntools.geospatial.tools import geocode # Just for this exercise



import folium 

from folium import Marker

from folium.plugins import MarkerCluster
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
# load and preview Starbucks locations in California

starbucks = pd.read_csv("../input/geospatial-learn-course-data/starbucks_locations.csv")

starbucks.head()
# how many rows in each column have missing values?

print(starbucks.isnull().sum())
# rows with missing locations

rows_with_missing = starbucks[starbucks['Longitude'].isnull() | starbucks['Latitude'].isnull()]

rows_with_missing
# define geocoder function

def my_geocoder(row):

    try:

        point = geocode(row, provider='nominatim').geometry[0]

        return pd.Series({'Longitude': point.x, 'Latitude': point.y})

    except:

        return None



# fill missing geo data

rows_with_missing = rows_with_missing.apply(lambda x: my_geocoder(x['Address']), axis=1)



# drop rows that were not successfully geocoded

rows_with_missing.dropna(axis=0, subset=['Longitude', 'Latitude'])



# update main DataFrame

starbucks.update(rows_with_missing)



# check your answer

q_1.check()
# line below will give you solution code

# q_1.solution()
# create a base map

m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)



# add a marker for each Berkeley location

for idx, row in starbucks[starbucks["City"]=='Berkeley'].iterrows():

    Marker([row['Latitude'], row['Longitude']], popup=row['Store Name']).add_to(m_2)



# uncomment to see a hint

# q_2.a.hint()



# show the map

embed_map(m_2, 'q_2.html')
# get credit for your work after you have created a map

q_2.a.check()



# uncomment to see our solution (your code may look different!)

# q_2.a.solution()
# view the solution (Run this code cell to receive credit!)

q_2.b.solution()
CA_counties = gpd.read_file("../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp")

CA_counties.head()
CA_pop = pd.read_csv("../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID")

CA_high_earners = pd.read_csv("../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID")

CA_median_age = pd.read_csv("../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID")
# solution with join + merge

cols_to_add = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()

CA_stats = CA_counties.merge(cols_to_add, on="GEOID")



# solution with multi merges

# CA_stats = CA_counties.merge(CA_pop, on="GEOID").merge(CA_high_earners, on="GEOID").merge(CA_median_age, on="GEOID")



# change CRS code

CA_stats.crs = {'init': 'epsg:4326'}



# check your answer

q_3.check()
# lines below will give you a hint or solution code

# q_3.hint()

# q_3.solution()
CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]
CA_stats.head()
sel_counties = CA_stats[

    (CA_stats['high_earners'] >= 100000) &

    (CA_stats['median_age'] <= 38.5) &

    (CA_stats['density'] >= 285) &

    (

        (CA_stats['high_earners'] >= 500000) |

        (CA_stats['median_age'] <= 35.5) |

        (CA_stats['density'] >= 1400)

    )

]



# check your answer

q_4.check()
# lines below will give you a hint or solution code

# q_4.hint()

# q_4.solution()
sel_counties
starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks['Longitude'], starbucks['Latitude']))

starbucks_gdf.crs = {'init': 'epsg:4326'}
sel_counties_stores = gpd.sjoin(starbucks_gdf, sel_counties)

num_stores = len(sel_counties_stores)



# check your answer

q_5.check()
# lines below will give you a hint or solution code

# q_5.hint()

# q_5.solution()
sel_counties_stores.head()
# Create a base map

m_6 = folium.Map(location=[37,-120], zoom_start=6)



# show selected store locations

import math

mc = MarkerCluster()

for idx, row in sel_counties_stores.iterrows():

    mc.add_child(Marker([row["Latitude"], row["Longitude"]]))



m_6.add_child(mc)



# uncomment to see a hint

# q_6.hint()



# show the map

embed_map(m_6, 'q_6.html')
# get credit for your work after you have created a map

q_6.check()



# uncomment to see our solution (your code may look different!)

# q_6.solution()