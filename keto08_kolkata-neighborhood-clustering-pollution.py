import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans


import folium # map rendering library

print('Libraries imported.')
address = 'Kolkata, IN'

geolocator = Nominatim(user_agent="kol_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Kolkata are {}, {}.'.format(latitude, longitude))
df = pd.read_csv("../input/locations.csv")
df.rename(columns={"icon":"Neighborhood"}, inplace = True) 
#df.drop(df.columns[3], axis = 1, inplace = True) 
df
# create map of Kolkata using latitude and longitude values
map_kolkata = folium.Map(location=[latitude, longitude], zoom_start=10.5)

# add markers to map
for lat, lng, neighborhood, pm in zip(df['latitude'], df['longitude'], df['Neighborhood'], df['PM2.5']):
    label = '{}, {}'.format(neighborhood, pm)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_kolkata)  
    
map_kolkata
CLIENT_ID = '1YIN3BTQWIJL3VGWHDWYT3ECVGNG0QEWSAMZ0S3QAOUXPGOY' # your Foursquare ID
CLIENT_SECRET = 'VBOXXITNIQNFPPQ0Y0N4OPWN4Q4ZOAERS3UYRWWX2YMGV4I0' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
neighborhood_latitude = df.loc[0, 'latitude'] # neighborhood latitude value
neighborhood_longitude = df.loc[0, 'longitude'] # neighborhood longitude value

neighborhood_name = df.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL
results = requests.get(url).json()
results
# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
kolkata_venues = getNearbyVenues(names=df['Neighborhood'],
                                   latitudes=df['latitude'],
                                   longitudes=df['longitude']
                                  )
print(kolkata_venues.shape)
kolkata_venues.head()
kolkata_venues.groupby('Neighborhood').count()
print('There are {} uniques categories.'.format(len(kolkata_venues['Venue Category'].unique())))
# one hot encoding
kolkata_onehot = pd.get_dummies(kolkata_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
kolkata_onehot['Neighborhood'] = kolkata_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [kolkata_onehot.columns[-1]] + list(kolkata_onehot.columns[:-1])
kolkata_onehot = kolkata_onehot[fixed_columns]

kolkata_onehot.head()
kolkata_onehot.shape
kolkata_grouped = kolkata_onehot.groupby('Neighborhood').mean().reset_index()
kolkata_grouped
kolkata_grouped.shape
num_top_venues = 5

for hood in kolkata_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = kolkata_grouped[kolkata_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = kolkata_grouped['Neighborhood']

for ind in np.arange(kolkata_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(kolkata_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
# set number of clusters
kclusters = 5

kolkata_grouped_clustering = kolkata_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(kolkata_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
kolkata_grouped_clustering
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

kolkata_merged = df

# merge kolkata_grouped with kolkata_data to add latitude/longitude for each neighborhood
kolkata_merged = kolkata_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

kolkata_merged.head() 
kolkata_merged
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10.5)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(kolkata_merged['latitude'], kolkata_merged['longitude'], kolkata_merged['Neighborhood'], kolkata_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
c0=kolkata_merged.loc[kolkata_merged['Cluster Labels'] == 0]
c0
c0['PM2.5'].mean()
c1 = kolkata_merged.loc[kolkata_merged['Cluster Labels'] == 1]
c1
c1['PM2.5'].mean()
c2 = kolkata_merged.loc[kolkata_merged['Cluster Labels'] == 2]
c2
c2['PM2.5'].mean()
c3 = kolkata_merged.loc[kolkata_merged['Cluster Labels'] == 3]
c3
c3['PM2.5'].mean()
c4 = kolkata_merged.loc[kolkata_merged['Cluster Labels'] == 4]
c4
c4['PM2.5'].mean()
result = [{'Cluster': 'Cluster 1', 'PM2.5': round(c0['PM2.5'].mean(),2)},\
          {'Cluster': 'Cluster 2', 'PM2.5': round(c1['PM2.5'].mean(),2) },\
          {'Cluster': 'Cluster 3', 'PM2.5': round(c2['PM2.5'].mean(),2) },\
          {'Cluster': 'Cluster 4', 'PM2.5': round(c3['PM2.5'].mean(),2) },\
          {'Cluster': 'Cluster 5', 'PM2.5': round(c4['PM2.5'].mean(),2) }]
result
result_df = pd.DataFrame(result) 
result_df
