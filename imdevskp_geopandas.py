# importing libraries



import pandas as pd



import matplotlib.pyplot as plt



import geopandas as gpd

from shapely.geometry import Point, LineString



import folium

from folium import Marker, GeoJson

from folium.plugins import MarkerCluster, HeatMap

wc = gpd.read_file('../input/human-development-index-hdi/countries.geojson')

wc.head(2)
wc.columns
wc.economy
fig, ax = plt.subplots(figsize=(10, 5))

wc.plot(ax=ax,color='midnightblue')

plt.show()
type(wc.geometry)
len(wc)
wc.geometry[:5]
wc.geometry[:5].area
# wc_json = wc.to_json()

# print(wc_json)
italy = wc[wc['name'] =='Italy']

italy.plot()
china = wc[wc['name'] =='China']

china.plot()
wc.plot(column='labelrank', cmap='Blues_r', figsize=(10, 5))
wc.plot(column='labelrank', cmap='Greens', figsize=(10, 5))
wc.plot(column='economy', cmap='Greens_r', figsize=(10, 5), legend=True)
na = wc[wc['continent']=='Asia']

na.plot(column='labelrank', cmap='Greens', legend=True, figsize=(10, 5))
leg_kwds={'title':'District Number',

          'loc': 'upper left',

          'bbox_to_anchor':(1, 1.03),

          'ncol':3}



na = wc[wc['continent']=='South America']

na.plot(column='admin', cmap='Set2', legend=True, legend_kwds=leg_kwds)
leg_kwds={'title':'District Number',

          'loc': 'upper left',

          'bbox_to_anchor':(1, 1.03),

          'ncol':4}



na = wc[wc['continent']=='Asia']

na.plot(column='admin', cmap='Set2', legend=True, legend_kwds=leg_kwds)
na.plot(column='labelrank', cmap='Reds', legend=True, scheme='equal_interval', k=2, figsize=(10, 5))
na.plot(column='labelrank', cmap='Reds', legend=True, scheme='equal_interval', k=4, figsize=(10, 5))
na.plot(column='labelrank', cmap='Reds', legend=True, scheme='quantiles', k=3, figsize=(10, 5))
na.plot(column='labelrank', cmap='Reds', legend=True, scheme='quantiles', k=3, figsize=(10, 5))
# ! ls ../input/natural-earth/110m_cultural/
# most populated cities

cities = gpd.read_file('../input/natural-earth/110m_cultural/ne_110m_populated_places.shp')

# cities.head(2)
fig, ax = plt.subplots(figsize=(12, 6))

wc.plot(ax=ax, color='lightgrey')

cities.plot(ax=ax, color='darkorange', markersize=10)

ax.set_axis_off()
brussels = cities.loc[170, 'geometry']

print(brussels)

print(type(brussels))

brussels
belgium = wc[wc['name']=='Belgium']['geometry'].squeeze()

uk = wc[wc['name']=='United Kingdom']['geometry'].squeeze()

germany = wc[wc['name']=='Germany']['geometry'].squeeze()

ireland = wc[wc['name']=='Ireland']['geometry'].squeeze()



gpd.GeoSeries([belgium, uk, germany, ireland]).plot()
# .crs
# to crs
# .area .centroid
brussels = cities.loc[170, 'geometry']

dublin = cities.loc[156, 'geometry']



brussels.distance(dublin)
belgium.contains(brussels)
ireland.contains(brussels)
brussels.within(belgium)
belgium.touches(germany)
belgium.touches(uk)
# creating line 

dublin_brussels_line = LineString(zip((brussels.x,dublin.x ), (brussels.y, dublin.y)))



fig, ax = plt.subplots()

gpd.GeoSeries([belgium, uk, germany, ireland]).plot(color='gainsboro', ax=ax)

gpd.GeoSeries([dublin_brussels_line]).plot(color='deeppink', ax = ax)

ax.set_axis_off()
for i in [belgium, uk, germany, ireland]:

    print(dublin_brussels_line.intersects(i))
rivers = gpd.read_file('../input/natural-earth/110m_physical/ne_110m_rivers_lake_centerlines.shp')

# rivers.head(2)
fig, ax = plt.subplots(figsize=(12, 6))

wc.plot(ax=ax, color='gainsboro')

rivers.plot(ax=ax, color='teal', markersize=10)

ax.set_axis_off()
amazon = rivers[rivers['name']=='Amazonas']

amazon
amazon.geometry
amazon.geometry.squeeze()
fig, ax = plt.subplots(figsize=(12, 6))

wc[wc['continent']=='South America'].plot(ax=ax, color='gainsboro')

rivers[rivers['name']=='Amazonas'].plot(ax=ax, color='teal', markersize=10)

ax.set_axis_off()
print(wc.shape)

mask = wc.intersects(amazon.geometry.squeeze())

wc[mask]
mask = wc.intersects(dublin)

wc[mask]
full_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_lands/DEC_lands/DEC_lands.shp")

full_data.head(2)
type(full_data)
data = full_data.loc[:, ["CLASS", "COUNTY", "geometry"]].copy()
# How many lands of each type are there?

data['CLASS'].value_counts()
# Select lands that fall under the "WILD FOREST" or "WILDERNESS" category

wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()

wild_lands.head()
wild_lands.plot()
wild_lands.geometry.head()
# Campsites in New York state (Point)

POI_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_pointsinterest/DEC_pointsinterest/Decptsofinterest.shp")

campsites = POI_data.loc[POI_data.ASSET=='PRIMITIVE CAMPSITE'].copy()



# Foot trails in New York state (LineString)

roads_trails = gpd.read_file("../input/geospatial-learn-course-data/DEC_roadstrails/DEC_roadstrails/Decroadstrails.shp")

trails = roads_trails.loc[roads_trails.ASSET=='FOOT TRAIL'].copy()



# County boundaries in New York state (Polygon)

counties = gpd.read_file("../input/geospatial-learn-course-data/NY_county_boundaries/NY_county_boundaries/NY_county_boundaries.shp")
ax = counties.plot(figsize=(10, 10), color='none', edgecolor='grey', zorder=3)

wild_lands.plot(color='teal', ax=ax)

campsites.plot(color='red', markersize=2, ax=ax)

trails.plot(color='black', markersize=1, ax=ax)
regions = gpd.read_file("../input/geospatial-learn-course-data/ghana/ghana/Regions/Map_of_Regions_in_Ghana.shp")

print(regions.crs)
# Create a DataFrame with health facilities in Ghana

facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")



# Convert the DataFrame to a GeoDataFrame

facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))



ax = regions.plot(figsize=(8,8), color='whitesmoke', linestyle=':', edgecolor='black')

facilities.plot(markersize=1, ax=ax)
# Set the coordinate reference system (CRS) to EPSG 4326

facilities.crs = {'init': 'epsg:4326'}



# Create a map

ax = regions.plot(figsize=(8,8), color='whitesmoke', linestyle=':', edgecolor='black')

facilities.to_crs(epsg=32630).plot(markersize=1, ax=ax)
# The "Latitude" and "Longitude" columns are unchanged

facilities.head()
# The "Latitude" and "Longitude" columns are unchanged

facilities.to_crs(epsg=32630).head()
# Load the data and print the first 5 rows

birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])

print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))

birds_df.head()
# Create the GeoDataFrame

birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))



# Set the CRS to {'init': 'epsg:4326'}

birds.crs = {'init' :'epsg:4326'}
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin(['North America', 'South America'])]

americas.head()
ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

birds.plot(ax=ax, markersize=10)
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

end_gdf.crs = {'init': 'epsg:4326'}
# Your code here

ax = americas.plot(figsize=(10, 10), color='white', linestyle=':', edgecolor='gray')



start_gdf.plot(ax=ax, color='red',  markersize=30)

path_gdf.plot(ax=ax, cmap='tab20b', linestyle='-', linewidth=1, zorder=1)

end_gdf.plot(ax=ax, color='black', markersize=30)

# Path of the shapefile to load

protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"



# Your code here



protected_areas = gpd.read_file(protected_filepath)



# Country boundaries in South America

south_america = americas.loc[americas['continent']=='South America']



# Your code here: plot protected areas in South America

ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')

protected_areas.plot(ax=ax, alpha=0.4)
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])



# Your code here: Calculate the total area of South America (in square kilometers)

totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6



# What percentage of South America is protected?

percentage_protected = P_Area/totalArea

print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
# Your code here

ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')

protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)

birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)
from geopandas.tools import geocode
geocode("Taj Mahal")
geocode("The White House")
result = geocode("The Great Pyramid of Giza", provider="nominatim")

result
point = result.geometry.iloc[0]

print("Latitude:", point.y)

print("Longitude:", point.x)
universities = pd.read_csv("../input/geospatial-learn-course-data/top_universities.csv")

universities.head()
import numpy as np



def my_geocoder(row):

    try:

        point = geocode(row, provider='nominatim').geometry.iloc[0]

        return pd.Series({'Latitude': point.y, 'Longitude': point.x, 'geometry': point})

    except:

        return None



universities[['Latitude', 'Longitude', 'geometry']] = universities.apply(lambda x: my_geocoder(x['Name']), axis=1)



print("{}% of addresses were geocoded!".format(

    (1 - sum(np.isnan(universities["Latitude"])) / len(universities)) * 100))



# Drop universities that were not successfully geocoded

universities = universities.loc[~np.isnan(universities["Latitude"])]

universities = gpd.GeoDataFrame(universities, geometry=universities.geometry)

universities.crs = {'init': 'epsg:4326'}

universities.head()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

europe = world.loc[world.continent == 'Europe'].reset_index(drop=True)



europe_stats = europe[["name", "pop_est", "gdp_md_est"]]

europe_boundaries = europe[["name", "geometry"]]
europe_boundaries.head()
europe_stats.head()
europe = europe_boundaries.merge(europe_stats, on="name")

europe.head()
# Use spatial join to match universities to countries in Europe

european_universities = gpd.sjoin(universities, europe)



# Investigate the result

print("We located {} universities.".format(len(universities)))

print("Only {} of the universities were located in Europe (in {} different countries).".format(

    len(european_universities), len(european_universities.name.unique())))



european_universities.head()
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



print(starbucks.isnull().sum())
# Create a base map

m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)



# Your code here: Add a marker for each Berkeley location

for idx, row in starbucks[starbucks["City"]=='Berkeley'].iterrows():

    Marker([row['Latitude'], row['Longitude']]).add_to(m_2)

    

# Show the map

m_2
CA_counties = gpd.read_file("../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp")

CA_pop = pd.read_csv("../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID")

CA_high_earners = pd.read_csv("../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID")

CA_median_age = pd.read_csv("../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID")
cols_to_add = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()

CA_stats = CA_counties.merge(cols_to_add, on="GEOID")
CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]
sel_counties = CA_stats[((CA_stats.high_earners > 100000) &

                         (CA_stats.median_age < 38.5) &

                         (CA_stats.density > 285) &

                         ((CA_stats.median_age < 35.5) |

                         (CA_stats.density > 1400) |

                         (CA_stats.high_earners > 500000)))]
starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks.Longitude, starbucks.Latitude))

starbucks_gdf.crs = {'init': 'epsg:4326'}
# Fill in your answer

locations_of_interest = gpd.sjoin(starbucks_gdf, sel_counties)

num_stores = len(locations_of_interest)
import math
# Create a base map

m_6 = folium.Map(location=[37,-120], zoom_start=6)



# Your code here: show selected store locations

mc = MarkerCluster()



locations_of_interest = gpd.sjoin(starbucks_gdf, sel_counties)

for idx, row in locations_of_interest.iterrows():

    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

        mc.add_child(folium.Marker([row['Latitude'], row['Longitude']]))



m_6.add_child(mc)



# Uncomment to see a hint

#q_6.hint()



# Show the map

m_6
releases = gpd.read_file("../input/geospatial-learn-course-data/toxic_release_pennsylvania/toxic_release_pennsylvania/toxic_release_pennsylvania.shp") 

releases.head()
stations = gpd.read_file("../input/geospatial-learn-course-data/PhillyHealth_Air_Monitoring_Stations/PhillyHealth_Air_Monitoring_Stations/PhillyHealth_Air_Monitoring_Stations.shp")

stations.head()
print(stations.crs)

print(releases.crs)
# Select one release incident in particular

recent_release = releases.iloc[360]



# Measure distance from release to each station

distances = stations.geometry.distance(recent_release.geometry)

distances
print('Mean distance to monitoring stations: {} feet'.format(distances.mean()))

print('Closest monitoring station ({} feet):'.format(distances.min()))

print(stations.iloc[distances.idxmin()][["ADDRESS", "LATITUDE", "LONGITUDE"]])
two_mile_buffer = stations.geometry.buffer(2*5280)

two_mile_buffer.head()
# Create map with release incidents and monitoring stations

m = folium.Map(location=[39.9526,-75.1652], zoom_start=11)

HeatMap(data=releases[['LATITUDE', 'LONGITUDE']], radius=15).add_to(m)

for idx, row in stations.iterrows():

    Marker([row['LATITUDE'], row['LONGITUDE']]).add_to(m)

    

# Plot each polygon on the map

GeoJson(two_mile_buffer.to_crs(epsg=4326)).add_to(m)



# Show the map

m
# Turn group of polygons into single multipolygon

my_union = two_mile_buffer.geometry.unary_union

print('Type:', type(my_union))



# Show the MultiPolygon object

my_union
# The closest station is less than two miles away

my_union.contains(releases.iloc[360].geometry)
# The closest station is more than two miles away

my_union.contains(releases.iloc[358].geometry)