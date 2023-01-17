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

    try:

        point = geocode(row, provider='nominatim').geometry.iloc[0]

        return pd.Series({'Latitude': point.y, 'Longitude': point.x, 'geometry': point})

    except:

        return None



rows_with_missing[['Latitude', 'Longitude', 'geometry']] = rows_with_missing.apply(lambda x: my_geocoder(x['Address']), axis=1)



# Check your answer

q_1.check()
rows_with_missing

starbucks.update(rows_with_missing)

starbucks.iloc[155]
# Line below will give you solution code

#q_1.solution()
# Create a base map

m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)



# Your code here: Add a marker for each Berkeley location

for idx, row in starbucks[starbucks["City"]=='Berkeley'].iterrows():

    Marker([row['Latitude'], row['Longitude']], popup=row['Store Name']).add_to(m_2)

    

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

CA_stats = CA_counties.merge(CA_pop, on = "GEOID").merge(CA_high_earners, on = "GEOID").merge(CA_median_age, on = "GEOID")



#cols_to_add = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()

#CA_stats = CA_counties.merge(cols_to_add, on="GEOID")



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]
CA_stats.head()
# Your code here

sel_counties = CA_stats[((CA_stats.high_earners > 100000) &

                         (CA_stats.median_age < 38.5) &

                         (CA_stats.density > 285) &

                         ((CA_stats.median_age < 35.5) |

                         (CA_stats.density > 1400) |

                         (CA_stats.high_earners > 500000)))]



# Check your answer

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()
starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks.Longitude, starbucks.Latitude))

starbucks_gdf.crs = {'init': 'epsg:4326'}
# Fill in your answer

num_stores = len(gpd.sjoin(sel_counties, starbucks_gdf))



# Check your answer

q_5.check()
# Lines below will give you a hint or solution code

#q_5.hint()

#q_5.solution()
# Create a base map

m_6 = folium.Map(location=[37,-120], zoom_start=6)



# Your code here: show selected store locations

for idx, row in gpd.sjoin(sel_counties, starbucks_gdf).iterrows():

    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

        Marker([row['Latitude'], row['Longitude']], popup=('High earners: $' + str(row['high_earners'])), parse_html=True).add_to(m_6)





# Uncomment to see a hint

#q_6.hint()



# Show the map

embed_map(m_6, 'q_6.html')
# Get credit for your work after you have created a map

q_6.check()



# Uncomment to see our solution (your code may look different!)

#q_6.solution()