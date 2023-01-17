#import need modules



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import folium

import requests # library to handle requests

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

from sklearn.cluster import DBSCAN

import matplotlib.cm as cm

import matplotlib.colors as colors
#set parameters for api call

client_id = 'THIIRDS3XMAMXQHJMABQD0FY3SGAU2VOKO5L5GHZUIMJXP4L' # your Foursquare ID

client_secret = '2SQU0R45T05O0ESS5JCBYIRZCRIPODIXBEG4VOSMM3KZ0LSF' # your Foursquare Secret

version = '20180605' # Foursquare API version



limit = '50'

intent = 'browse'

ne1 = '47.687772, -122.251590'

sw1 = '47.6, -122.430775'



ne2 = '47.599999, -122.251590'

sw2 = '47.531678, -122.430775'



brewery = '50327c8591d4c4b30a586d5d'

food = '4d4b7105d754a06374d81259'
#set map parameters



address = 'Seattle, WA'

geolocator = Nominatim(user_agent="sea_explorer")

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print('The geograpical coordinate of Seattle are {}, {}.'.format(latitude, longitude))
#define urls for calls - this is split into two to avoid the 50 row limitation in the api call



url1 = 'https://api.foursquare.com/v2/venues/search?client_id=%s&client_secret=%s&v=%s&intent=%s&ne=%s&sw=%s&categoryId=%s&limit=%s' % (client_id, client_secret, version, intent, ne1, sw1, brewery, limit)

url2 = 'https://api.foursquare.com/v2/venues/search?client_id=%s&client_secret=%s&v=%s&intent=%s&ne=%s&sw=%s&categoryId=%s&limit=%s' % (client_id, client_secret, version, intent, ne2, sw2, brewery, limit)
results1 = requests.get(url1).json()

results2 = requests.get(url2).json()
#process results into dataframes and then merge

results1_df = json_normalize(results1['response']['venues'])

results1_df = results1_df[['name', 'location.lat', 'location.lng']]



results2_df = json_normalize(results2['response']['venues'])

results2_df = results2_df[['name', 'location.lat', 'location.lng']]
df = pd.concat([results1_df, results2_df])

df=df.reset_index(drop=True)

df
# create map of Seattle using latitude and longitude values

map_seattle = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to map

for lat, lng, name in zip(df['location.lat'], df['location.lng'], df['name']):

    label = '{}'.format(name)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_seattle)  

    

map_seattle
X = df[['location.lat', 'location.lng']]
#initialize dbscan

epsilon = 0.01

minimumSamples = 4

db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)

labels = db.labels_

labels
df.insert(0, 'Cluster Labels', labels)
df
clusters = df['Cluster Labels'].unique()

clusters
# create map of Seattle using latitude and longitude values

cluster_map = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters



def label_color(label):

    if label == -1:

        return 'grey'

    elif label == 0:

        return 'yellow',

    elif label == 1:

        return 'green',

    elif label == 2:

        return 'blue',

    elif label == 3:

        return 'red',

    elif label == 4:

        return 'pink',

    elif label == 5:

        return 'orange',

    else:

        return 'black'



# add markers to map

markers_colors = []

for lat, lng, name, cluster in zip(df['location.lat'], df['location.lng'], df['name'], df['Cluster Labels']):

    label = folium.Popup(str(name) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color=label_color(cluster),

        fill=True,

        fill_color=label_color(cluster),

        fill_opacity=0.7).add_to(cluster_map)  

    

cluster_map
cluster_df = df.groupby(['Cluster Labels']).mean()

cluster_df
radius = '100'
#determine proximity of food locations

for x in range (0,6):

    record = cluster_df.loc[x]

    lat = record['location.lat']

    long = record['location.lng']

    url3 = 'https://api.foursquare.com/v2/venues/search?client_id=%s&client_secret=%s&v=%s&intent=%s&ll=%s,%s&radius=%s&categoryId=%s&limit=%s' % (client_id, client_secret, version, intent, lat, long, radius, food, limit)

    results3 = requests.get(url3).json()

    results3_df = json_normalize(results3['response']['venues'])

    print (str(x) + ' ' + str(results3_df.shape[0]))