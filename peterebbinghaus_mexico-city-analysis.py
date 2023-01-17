import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
CLIENT_ID = 'hidden' # your Foursquare ID
CLIENT_SECRET = 'hidden' # your Foursquare Secret
VERSION = '20200723' # Foursquare API version

#print('Your credentails:')
#print('CLIENT_ID: ' + CLIENT_ID)
#print('CLIENT_SECRET:' + CLIENT_SECRET)
latitude = 19.42847
longitude =  -99.12766
radius = 50000
LIMIT = 1000
# type your answer here
url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)
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

df=nearby_venues

df.head()
print('{} venues were returned by Foursquare.'.format(df.shape[0]))
# create map of Toronto using latitude and longitude values
map_cdmx = folium.Map(location=[latitude, longitude], zoom_start=13)
# add markers to map
for la, ln, borough, neighborhood in zip(df['lat'], df['lng'], df['categories'], df['name']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [la, ln],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_cdmx)  
    
map_cdmx
df.groupby("categories").count()["name"].sort_values(ascending=False)
print('There are {} uniques categories.'.format(len(df['categories'].unique())))
# one hot encoding
onehot = pd.get_dummies(df[['categories']], prefix="", prefix_sep="")
onehot
# add neighborhood column back to dataframe
onehot['name'] = df['name'] 

# move neighborhood column to the first column
fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
onehot = onehot[fixed_columns]

onehot.head()
onehot.shape
# set number of clusters
kclusters = 10

clustering = onehot.drop('name', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# add clustering labels
df.insert(0, 'Cluster Labels', kmeans.labels_)

#merged = df

# merge grouped with df to add latitude/longitude for each restaurant name
#merged = merged.join(df.set_index('name'), on='name')


df.tail() # check the last columns!
df.describe()
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=13)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df['lat'], df['lng'], df['categories'], df['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
df.loc[df['Cluster Labels'] == 0, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 1, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 2, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 3, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 4, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 5, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 6, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 7, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 8, df.columns[[2] + list(range(5, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 9, df.columns[[2] + list(range(5, df.shape[1]))]]