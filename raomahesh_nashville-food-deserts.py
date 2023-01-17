# Standard data analysis packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Working with geospatial data
import geopandas as gpd
import folium
from shapely.ops import unary_union
from sklearn.metrics.pairwise import haversine_distances

# Webscraping and APIs
import requests
from bs4 import BeautifulSoup as BS

# Additional python packages
import json
import math
import ast

%matplotlib inline
tennessee_tracts = gpd.read_file('../input/nashville-food-deserts-data/cb_2018_47_tract_500k.shp')
tennessee_tracts.head()
# Save the first row POLYGON to a variable
single_tract = tennessee_tracts.loc[0, 'geometry']

# print the type of the object
print('object type: ',type(single_tract), '\n')

# print what is stored inside the object
print('object contents: ', single_tract)
tennessee_tracts.boundary.plot(figsize = (25, 5));
# Select only Davidson County tracts (COUNTYFP == 037)
davidson_tracts = tennessee_tracts[tennessee_tracts['COUNTYFP']=='037'].reset_index(drop = True)
davidson_tracts.head()
davidson_tracts.boundary.plot(figsize = (20, 20));
davidson_county = gpd.read_file('../input/nashville-food-deserts-data/Davidson County Border (GIS).geojson')
davidson_county.boundary.plot(figsize = (20, 20));
# modified from https://censusreporter.org/data/table/?table=B17001&geo_ids=14000US47037016000,05000US47037,04000US47,01000US,140|05000US47037&primary_geo_id=14000US47037016000
poverty_rates_davidson = pd.read_csv('../input/nashville-food-deserts-data/davidson_poverty_cleaned.csv', index_col = 0)
poverty_rates_davidson.head()
# remove aggregation rows
poverty_rates_davidson_tract = poverty_rates_davidson.drop([0, 1, 2])
poverty_rates_davidson_tract.head()
poverty_rates_davidson_tract['pct_below_poverty_level'] = poverty_rates_davidson_tract['Income in the past 12 months below poverty level:']/poverty_rates_davidson_tract['Total:']
poverty_rates_davidson_tract.head()

poverty_rates_davidson_tract['GEOID'] = poverty_rates_davidson_tract['geoid'].str[7:]
davidson_tracts = davidson_tracts.merge(poverty_rates_davidson_tract[['GEOID', 'pct_below_poverty_level']],
                                        how = 'left',
                                        on = 'GEOID')

davidson_tracts['pct_below_poverty_level'] = davidson_tracts['pct_below_poverty_level'].fillna(np.median(davidson_tracts['pct_below_poverty_level']))
davidson_tracts['above_20_pct'] = (davidson_tracts['pct_below_poverty_level'] >= 0.2).astype(int)
davidson_tracts.head()
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.plot(ax=ax, column = 'above_20_pct', cmap='Set1', edgecolor = 'blue', alpha = 0.5)
ax = davidson_county.boundary.plot(ax=ax, color='blue');
davidson_service_districts = gpd.read_file('../input/nashville-food-deserts-data/Service Districts (GIS).geojson')
davidson_service_districts
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.plot(ax=ax, column = 'above_20_pct', cmap='Set1', edgecolor = 'blue', alpha = 0.5)
ax = davidson_county.boundary.plot(ax=ax, color='blue')
ax = davidson_service_districts[davidson_service_districts['name']=='Urban Services District'].plot(ax=ax, alpha = 0.4, edgecolor = 'black', linewidth = 3);
urban_service_area = davidson_service_districts.loc[0, 'geometry']
def check_urban_tract(row):
    return (row['geometry'] - urban_service_area).area/row['geometry'].area
davidson_tracts['rural_ratio'] = davidson_tracts.apply(check_urban_tract, axis = 1) # axis = 1 means apply to rows

davidson_tracts['is_rural'] = (davidson_tracts['rural_ratio'] > 0.5).astype(int)
davidson_tracts.head()
davidson_county_buffer_10 = gpd.GeoDataFrame(geometry = davidson_county['geometry'].buffer(0.14492753623188406))
davidson_county_buffer_10.plot();
davidson_county_buffer_1 = gpd.GeoDataFrame(geometry = davidson_county['geometry'].buffer(0.014492753623188406))
davidson_county_buffer_1.plot();
stores_gdf = gpd.read_file('../input/nashville-food-deserts-data/google_api_stores_cleaned.shp')
stores_gdf.head()
supermarkets = stores_gdf[stores_gdf['types'].apply(lambda x: 'supermarket' in x)]
supermarkets
non_dollar_supermarkets = supermarkets[~supermarkets['name'].str.contains('Dollar')]
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.plot(ax=ax, column = 'above_20_pct', categorical = True, cmap = 'Set1', edgecolor = '#377eb8', alpha = 0.5)
ax = davidson_county.boundary.plot(ax=ax, color='#377eb8')
non_dollar_supermarkets.plot(ax=ax, marker='o', color='#4e2853', markersize=15);
url = 'https://www.picktnproducts.org/listview/farmers-market.html'
response = requests.get(url)
soup = BS(response.content, 'lxml')
markets_names_soup = soup.find('div', attrs = {'id': 'middle'})
markets_names_soup
json.loads(markets_names_soup['data-middlejson'])
mid_tn_farm_mark = pd.DataFrame(json.loads(markets_names_soup['data-middlejson']))
mid_tn_farm_mark.head()
mid_tn_farm_mark = gpd.GeoDataFrame(mid_tn_farm_mark,
                                    geometry = gpd.points_from_xy(mid_tn_farm_mark['longitude'], mid_tn_farm_mark['latitude']))

keep_points_fm = []
rural_only_fm = []
for ind, p in mid_tn_farm_mark['geometry'].iteritems():
    if p.within(davidson_county_buffer_10.loc[0, 'geometry']):
        keep_points_fm.append(ind)
        if p.within(davidson_county_buffer_1.loc[0, 'geometry']):
            rural_only_fm.append(0)
        else:
            rural_only_fm.append(1)

davidson_farm_mark = mid_tn_farm_mark.loc[keep_points_fm]
davidson_farm_mark['rural_only'] = rural_only_fm

davidson_farm_mark.head()
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.plot(ax=ax, column = 'above_20_pct', categorical = True, cmap = 'Set1', edgecolor = '#377eb8', alpha = 0.5)
ax = davidson_county.boundary.plot(ax=ax, color='#377eb8')
ax = non_dollar_supermarkets.plot(ax=ax, marker='o', color='#4e2853', markersize=15)
davidson_farm_mark.plot(ax=ax, marker='o', color='#dddd19', markersize=15);
urban_store_buffers = gpd.GeoDataFrame(geometry = non_dollar_supermarkets[non_dollar_supermarkets['rural_only']==0]['geometry'].buffer(0.014492753623188406))


rural_store_buffers = gpd.GeoDataFrame(geometry = non_dollar_supermarkets['geometry'].buffer(0.14492753623188406))
# Add in Farmers Markets data

urban_store_buffers = urban_store_buffers.append(
    gpd.GeoDataFrame(
        geometry = davidson_farm_mark[davidson_farm_mark['rural_only']==0]['geometry'].buffer(0.014492753623188406)
    ))

rural_store_buffers = rural_store_buffers.append(
    gpd.GeoDataFrame(
        geometry = davidson_farm_mark['geometry'].buffer(0.14492753623188406)
    ))
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.plot(ax=ax, column = 'above_20_pct', categorical = True, cmap = 'Set1', edgecolor = '#377eb8', alpha = 0.5)
ax = davidson_county.boundary.plot(ax=ax, color='#377eb8')
ax = urban_store_buffers.plot(ax=ax, color = '#5f9ea0', alpha = 0.2)
ax = non_dollar_supermarkets.plot(ax=ax, marker='o', color='#4e2853', markersize=15)
davidson_farm_mark.plot(ax=ax, marker='o', color='#dddd19', markersize=15);
urban_store_buffers = unary_union(urban_store_buffers['geometry'])
rural_store_buffers = unary_union(rural_store_buffers['geometry'])
def check_low_food_access(row):
    if row['is_rural'] == 1:
        return (row['geometry'] - rural_store_buffers).area/row['geometry'].area
    else:
        return (row['geometry'] - urban_store_buffers).area/row['geometry'].area
# check tract on row 2 of census tract df
check_low_food_access(davidson_tracts.loc[1, :])
davidson_tracts['ratio_low_food_access'] = davidson_tracts.apply(check_low_food_access, axis = 1)

davidson_tracts['possible_food_desert'] = ((davidson_tracts['ratio_low_food_access'] > 0.33) & (davidson_tracts['above_20_pct'] == 1)).astype(int)
davidson_tracts.head()
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.plot(ax=ax, column = 'possible_food_desert', categorical = True, cmap = 'Set1', edgecolor = '#377eb8', alpha = 0.5)
ax = davidson_county.boundary.plot(ax=ax, color='#377eb8')
ax = davidson_service_districts[davidson_service_districts['name']=='Urban Services District'].plot(ax=ax, alpha = 0.2, edgecolor = 'black', linewidth = 3)
ax = non_dollar_supermarkets.plot(ax=ax, marker='o', color='#4e2853', markersize=15)
davidson_farm_mark.plot(ax=ax, marker='o', color='#dddd19', markersize=15);
possible_food_deserts = unary_union(davidson_tracts[davidson_tracts['possible_food_desert']==1]['geometry'])
nash_map = folium.Map(location =  [36.1612, -86.7775], zoom_start = 11)
folium.GeoJson(possible_food_deserts).add_to(nash_map)
nash_map
# Find total geographic bounds of search area

xmin,ymin,xmax,ymax = davidson_county_buffer_10.total_bounds
# Find total geographic bounds of Davidson County

xmin_d,ymin_d,xmax_d,ymax_d = davidson_county.total_bounds

# Divide the length and width by 10 to get the increment between each grid point

x_increment = (xmax_d-xmin_d)/10
y_increment = (ymax_d-ymin_d)/10
# determine x coordinate values for grid points
grid_x_boundaries = [xmin]
new_bound = xmin
for i in range(int((xmax-xmin)/x_increment)+1):
    new_bound = new_bound + x_increment
    grid_x_boundaries.append(new_bound)
    
# determine x coordinate values for grid points
grid_y_boundaries = [ymin]
new_bound = ymin
for i in range(int((ymax-ymin)/y_increment)+1):
    new_bound = new_bound + y_increment
    grid_y_boundaries.append(new_bound)
# get list of all lats and lons across all grid points
lons = []
lats = []
for left, right in zip(grid_x_boundaries[:-1], grid_x_boundaries[1:]):
    for top, bottom in zip(grid_y_boundaries[:-1], grid_y_boundaries[1:]):
        lats.append((top+bottom)/2)
        lons.append((left+right)/2)
        
# Take each pair of longitude and latitude, combine them, and convert to point object
grid_points = gpd.points_from_xy(lons, lats)

# put into geodataframe
grid_gdf = gpd.GeoDataFrame(geometry = grid_points)
grid_gdf.plot();
# Only keep points within the buffered Davidson county polygon
keep_points = []
for ind, p in grid_gdf['geometry'].iteritems():
    if p.within(davidson_county_buffer_10.loc[0, 'geometry']) or p.within(davidson_county_buffer_10.loc[1, 'geometry']):
        keep_points.append(ind)

grid_points_sub = grid_gdf.loc[keep_points].reset_index(drop=True)
base = davidson_county_buffer_10.plot(color='white', edgecolor='black')

grid_points_sub.plot(ax=base, marker='o', color='red', markersize=5);
# function adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
def dist_in_meters(point_1, point_2):
    point_1 = [math.radians(l) for l in [point_1.y, point_1.x]]
    point_2 = [math.radians(l) for l in [point_2.y, point_2.x]]
    dist_array_m = haversine_distances([point_1, point_2])*6371000
    return dist_array_m[0][1]
grid_point_radius = dist_in_meters(grid_points_sub.loc[1, 'geometry'], grid_points_sub.loc[2, 'geometry'])
grid_point_radius_mile = 3.0258050367212114828/69
grid_points_sub_buffers = gpd.GeoDataFrame(geometry = grid_points_sub['geometry'].buffer(grid_point_radius_mile))
f, ax = plt.subplots(1, figsize=(25, 25))
ax = davidson_tracts.boundary.plot(ax=ax, edgecolor = 'blue', color='blue', alpha = 0.15)
ax = davidson_county.boundary.plot(ax=ax, color='blue', alpha = 0.15)
ax = grid_points_sub_buffers.plot(ax=ax, color = '#ff7f00', alpha = 0.1)
grid_points_sub.plot(ax=ax, marker='o', color='#4e2853', markersize=5);
all_types = ['bakery',
             'convenience_store',
             'department_store',
             'drugstore',
             'gas_station',
             'grocery_or_supermarket',
             'home_goods_store',
             'supermarket',
             'pharmacy']
# Modified from https://python.gotrained.com/google-places-api-extracting-location-data-reviews/

def search_places_by_coordinate(location, radius, types, api_key, sleep_sec = 2):
    '''
    Send request to nearbysearch Google Maps endpoint
    
    location: The lat and lng to search nearby, "lat, lng"
    radius: The distance, in meters, to search around the location
    types: List of business types to search for
    api_key: Credentials provided by Google to authenticate use of the API
    sleep_sec: Number of seconds to wait between individual requests to throttle and avoid quotas
    '''
    # This is the endpoint where the request will be sent
    endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    places = [] # Where the responses will be saved
    
    # Formatting the request inputs
    params = {
        'location': location,
        'radius': radius,
        'types': types,
        'key': api_key
    }
    
    # Make the request to the endpoint with the associated parameters and save the output
    res = requests.get(endpoint_url, params = params)
    
    # Read the contents of the response, which is a json, into a dictionary to make it easier to work with
    results = json.loads(res.content)
    
    # Add the results to the any previous results
    places.extend(results['results'])
    
    # Wait before the next request
    time.sleep(sleep_sec)
    
    # If there are still more items available, the response will contain a next_page_token to pick up at the same spot
    # As long as there is a next_page_token, keep making requests
    while "next_page_token" in results:
        params['pagetoken'] = results['next_page_token'],
        res = requests.get(endpoint_url, params = params)
        results = json.loads(res.content)
        places.extend(results['results'])
        time.sleep(sleep_sec)
    
    # Once there are no more next_page_tokens, return the full list of stores
    return places
# creating output list in separate cell in case need to run for loop multiple times because of time out errors
responses = []
# This one can take a while, run with caution

for ind_2, t in enumerate(all_types):
    print(ind_2, t) # just to keep track of progress
    # if ind_2 >= 1: # uncomment and tab below over if need to start later in all_types list
    for ind, (lng, lat) in enumerate(list(zip(grid_points_sub['geometry'].x, grid_points_sub['geometry'].y))): # note that lat and lng are switched
        print(ind, lat, lng) # again, to keep track of progress
        # if ind >= 0: # uncomment and tab below if need to start later in grid df
        location = '{}, {}'.format(lat, lng)
        responses.append(search_places_by_coordinate(location, grid_point_radius, t, api_key))
stores_df = pd.read_csv('../input/nashville-food-deserts-data/google_api_stores_responses_9-4-20_2.csv')
stores_df.head()
stores_df['place_id'].value_counts()
stores_df = stores_df.drop_duplicates('place_id')
stores_df['place_id'].value_counts()
store_location_example = stores_df.loc[0, 'geometry']
print(store_location_example, '\n')
print(type(store_location_example)) # Double check the data type
store_location_example = ast.literal_eval(store_location_example)
print(type(store_location_example), '\n')
print(store_location_example)
lat_lng_example = store_location_example['location']
print(lat_lng_example, '\n')
pd.Series(lat_lng_example)
def extract_lat_lng_to_new_col(row):
    geo = row['geometry']
    geo = ast.literal_eval(geo)
    lat_lng = geo['location']
    lat_lng_s = pd.Series(lat_lng)
    row = row.append(lat_lng_s)
    return row
# axis = 1 ensures we are applying the function to each row, not each column
stores_df = stores_df.apply(extract_lat_lng_to_new_col, axis = 1)
stores_df.head()
stores_df = stores_df.rename(columns = {'geometry': 'geometry_google'})

stores_gdf = gpd.GeoDataFrame(stores_df,
                              geometry = gpd.points_from_xy(stores_df['lng'], # Creating the geometry column
                                                            stores_df['lat']))# on the fly with points_from_xy()

stores_gdf.head()
keep_stores = []
rural_only = []
for ind, p in stores_gdf['geometry'].iteritems(): # iteritems gives each index and value as a tuple
    # select stores to keep
    if p.within(davidson_county_buffer_10.loc[0, 'geometry']) or p.within(davidson_county_buffer_10.loc[1, 'geometry']):
        keep_stores.append(ind)
        # indicate if store is rural only or both
        if p.within(davidson_county_buffer_1.loc[0, 'geometry']) or p.within(davidson_county_buffer_1.loc[1, 'geometry']):
            rural_only.append(0)
        else:
            rural_only.append(1)

stores_gdf = stores_gdf.loc[keep_stores]
stores_gdf['rural_only'] = rural_only
stores_gdf.head()