import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

!pip install BeautifulSoup4
from bs4 import BeautifulSoup
import requests

import json # library to handle JSON files

!pip install geopy 
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
#!pip install -U scikit-learn scipy matplotlib
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('\nAll libraries imported successfully..!')
df = pd.read_csv('../input/wardwise-data-of-aurangabad-mh-india/Aurangabad Ward-Wise Data.csv')
df.head()
# get the coordinates of Aurangabad
address = 'Aurangabad, India'

geolocator = Nominatim(user_agent="my-application")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Aurangabad, India {}, {}.'.format(latitude, longitude))
# create map of Aurangabad using latitude and longitude values
map_abd = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_abd)  
    
map_abd
# define Foursquare Credentials and Version
CLIENT_ID = 'AANKWDPSR3JCT4TJY4OF5FNGFEDA5FMD4EURTDQUZHWUYKCS' # your Foursquare ID
CLIENT_SECRET = 'GM4FQDLRXMC4GMSG025IDFQNY1MCVUHW3ADL0CZOTDNNK4S2' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
radius = 5000
LIMIT = 100

venues = []

for lat, long, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    
    # create the API request URL
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        long,
        radius, 
        LIMIT)
    
    # make the GET request
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    # return only relevant information for each nearby venue
    for venue in results:
        venues.append((
            neighborhood,
            lat, 
            long, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))
# convert the venues list into a new DataFrame
venues_df = pd.DataFrame(venues)

# define the column names
venues_df.columns = ['Neighborhood', 'Latitude', 'Longitude', 'VenueName', 'VenueLatitude', 'VenueLongitude', 'VenueCategory']

print(venues_df.shape)
venues_df.head()
venues_df.groupby(["Neighborhood"]).count().head()
print('There are {} uniques categories.'.format(len(venues_df['VenueCategory'].unique())))
# print out the list of categories
venues_df['VenueCategory'].unique()
col=["Category"]
category_df = pd.DataFrame(data = venues_df['VenueCategory'].unique(),columns=col)
category_df.head()
# one hot encoding
abd_onehot = pd.get_dummies(venues_df[['VenueCategory']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
abd_onehot['Neighborhoods'] = venues_df['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [abd_onehot.columns[-1]] + list(abd_onehot.columns[:-1])
abd_onehot = abd_onehot[fixed_columns]

print(abd_onehot.shape)
abd_onehot.head()
abd_grouped = abd_onehot.groupby(["Neighborhoods"]).mean().reset_index()

print(abd_grouped.shape)
abd_grouped.head()
len(abd_grouped[abd_grouped["Restaurant"] > 0])
abd_restaurant = abd_grouped[["Neighborhoods","Restaurant"]]
abd_restaurant.head()
# set number of clusters
kclusters = 5

abd_clustering = abd_restaurant.drop(["Neighborhoods"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(abd_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
abd_merged = abd_restaurant.copy()

# add clustering labels
abd_merged["Cluster Labels"] = kmeans.labels_

abd_merged.rename(columns={"Neighborhoods": "Neighborhood"}, inplace=True)
abd_merged.head()
abd_merged = abd_merged.join(df.set_index("Neighborhood"), on="Neighborhood")

print(abd_merged.shape)
abd_merged.head() 
print(abd_merged.shape)
abd_merged.sort_values(["Cluster Labels"], inplace=True)
abd_merged.head()
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(abd_merged['Latitude'], abd_merged['Longitude'], abd_merged['Neighborhood'], abd_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
cluster0 = abd_merged.loc[abd_merged['Cluster Labels'] == 0].copy()
print('Number of Neighbourhoods:',cluster0.shape[0])
cluster0.head()
cluster1 = abd_merged.loc[abd_merged['Cluster Labels'] == 1].copy()
print('Number of Neighbourhoods:',cluster1.shape[0])
cluster1.head()
cluster2 = abd_merged.loc[abd_merged['Cluster Labels'] == 2].copy()
print('Number of Neighbourhoods:',cluster2.shape[0])
cluster2.head()
cluster3 = abd_merged.loc[abd_merged['Cluster Labels'] == 3].copy()
print('Number of Neighbourhoods:',cluster3.shape[0])
cluster3.head()
cluster4 = abd_merged.loc[abd_merged['Cluster Labels'] == 4].copy()
print('Number of Neighbourhoods:',cluster4.shape[0])
cluster4.head()
clusts = [cluster0,cluster1,cluster2,cluster3,cluster4]
mean_res = []
i=0
for c in clusts:
    mean_res.append([i,np.round(c['Restaurant'].mean(),4)])
    i=i+1
mean_res
col = ['Cluster','Mean Result']
res_mean_df = pd.DataFrame(data=mean_res,columns=col).set_index('Cluster')
res_mean_df
df_result = abd_merged.loc[abd_merged['Cluster Labels'] == 4].copy().reset_index(drop=True)
df_result.head()
df_result2 = df_result.sort_values(['Restaurant','Avg. Price'],ascending=[0,1])
df_final = df_result2
df_final.shape
df_final.head()
# create map
map_result = folium.Map(location=[latitude, longitude], zoom_start=12.5)

# add markers to the map
for lat, lon, poi, ward in zip(df_final['Latitude'], df_final['Longitude'], df_final['Neighborhood'], df_final['Ward']):
    label = folium.Popup(str(poi) +' '+ str(ward), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        #color=blue,
        fill=True,
        #fill_color=yellow,
        fill_opacity=0.7).add_to(map_result)
       
map_result
df_final_10 = df_final.head(10)

# create map
map_result = folium.Map(location=[latitude, longitude], zoom_start=12.5)

# add markers to the map
for lat, lon, poi, ward in zip(df_final_10['Latitude'], df_final_10['Longitude'], df_final_10['Neighborhood'], df_final_10['Ward']):
    label = folium.Popup(str(poi) +' '+ str(ward), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        #color=blue,
        fill=True,
        #fill_color=yellow,
        fill_opacity=0.7).add_to(map_result)
       
map_result