#Importing required libraries

import numpy as np

import pandas as pd



from geopy.geocoders import Nominatim

try:

    import geocoder

except:

    !pip install geocoder

    import geocoder



import requests

from bs4 import BeautifulSoup



try:

    import folium

except:

    !pip install folium

    import folium

    

from sklearn.cluster import KMeans
g = geocoder.arcgis('Bangalore, India')

blr_lat = g.latlng[0]

blr_lng = g.latlng[1]

print("The Latitude and Longitude of Bangalore is {} and {}".format(blr_lat, blr_lng))
#Scraping the webpage for list of localities

neig = requests.get("https://commons.wikimedia.org/wiki/Category:Suburbs_of_Bangalore").text
soup = BeautifulSoup(neig, 'html.parser')

#Creating a list to store neighborhood data

neighborhoodlist = []

for i in soup.find_all('div', class_='mw-category')[0].find_all('a'):

    neighborhoodlist.append(i.text)



#Creating a dataframe from the list

neig_df = pd.DataFrame({"Locality": neighborhoodlist})

neig_df.head()
#Shape of dataframe neig_df

neig_df.shape
#Defining a function to get the location of the localities

def get_location(localities):

    g = geocoder.arcgis('{}, Bangalore, India'.format(localities))

    get_latlng = g.latlng

    return get_latlng
co_ordinates = []

for i in neig_df["Locality"].tolist():

    co_ordinates.append(get_location(i))

print(co_ordinates)
#Creating a dataframe from the list of location

co_ordinates_df = pd.DataFrame(co_ordinates, columns=['Latitudes', 'Longitudes'])
#Adding co-ordinated to neig_df dataframe

neig_df["Latitudes"] = co_ordinates_df["Latitudes"]

neig_df["Longitudes"] = co_ordinates_df["Longitudes"]
neig_df.head()
#Creating a map

blr_map = folium.Map(location=[blr_lat, blr_lng],zoom_start=11)



#adding markers to the map for localities

#marker for Bangalore

folium.Marker([blr_lat, blr_lng], popup='<i>Bangalore</i>', color='red', tooltip="Click to see").add_to(blr_map)



#markers for localities

for latitude,longitude,name in zip(neig_df["Latitudes"], neig_df["Longitudes"], neig_df["Locality"]):

    folium.CircleMarker(

        [latitude, longitude],

        radius=6,

        color='blue',

        popup=name,

        fill=True,

        fill_color='#3186ff'

    ).add_to(blr_map)



blr_map
#Foursquare Credentials

# @hidden_cell

CLIENT_ID = 'JTB4R2ZERJU1QIVN1L4DXTEHZZS3ALDRVPDITI5KSV45D0DG'

CLIENT_SECRET = 'ICQ5C1WJOIFWHALH01K3XKDN4UFX3Q5PT3I4ZBNVW3P1SVKD'

VERSION = '20180605' # Foursquare API version



print('Your credentails:')

print('CLIENT_ID: ' + "CLIENT_ID")

print('CLIENT_SECRET:' + "CLIENT_SECRET")
#Getting the top 100 venues in each locality

radius = 2000

LIMIT = 100



venues = []



for lat, lng, locality in zip(neig_df["Latitudes"], neig_df["Longitudes"], neig_df["Locality"]):

    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, lat, lng, VERSION, radius, LIMIT)

    results = requests.get(url).json()['response']['groups'][0]['items']



    for venue in results:

        venues.append((locality, lat, lng, venue['venue']['name'], venue['venue']['location']['lat'], venue['venue']['location']['lng'], venue['venue']['categories'][0]['name']))
venues[0]
#Convert the venue list into dataframe

venues_df = pd.DataFrame(venues)

venues_df.columns = ['Locality', 'Latitude', 'Longitude', 'Venue name', 'Venue Lat', 'Venue Lng', 'Venue Category']

venues_df.head()
#Number of venues for each Locality

venues_df.groupby(['Locality']).count()
#Getting the unique categories

print('There are {} unique categries.'.format(len(venues_df['Venue Category'])))
#List of categories

print('Total number of unique catefories are {}'.format(len(venues_df['Venue Category'].unique().tolist())))

#First 10 categories

venues_df['Venue Category'].unique().tolist()#[:10]
#one hot encoding

blr_onehot = pd.get_dummies(venues_df[['Venue Category']], prefix="", prefix_sep="")



blr_onehot['Locality'] = venues_df['Locality']



#move the locality column to the front

blr_onehot = blr_onehot[ [ 'Locality' ] + [ col for col in blr_onehot.columns if col!='Locality' ] ]

blr_onehot.head()
blr_grouped = blr_onehot.groupby(['Locality']).mean().reset_index()

print(blr_grouped.shape)

blr_grouped.head()
#numbers of localities having Italian Restaurants

len(blr_grouped[blr_grouped['Italian Restaurant'] > 0])
blr_italian = blr_grouped[['Locality', 'Italian Restaurant']]

blr_italian.head()
#K-means clustering

cluster = 3 



#Dataframe for clustering

blr_clustering = blr_italian.drop(['Locality'], 1)



#run K-means clustering

k_means = KMeans(init="k-means++", n_clusters=cluster, n_init=12).fit(blr_clustering)



#getting the labels for first 10 locality 

print(k_means.labels_[0:10])
#Creating a of blr_italian dataframe

blr_labels = blr_italian.copy()



#addring label to blr_labels

blr_labels["Cluster Label"] = k_means.labels_



blr_labels.head()
#Merging the blr_labels and neig_df dataframes to get the latitude and longitudes for each locality

blr_labels = blr_labels.join(neig_df.set_index('Locality'), on='Locality')

blr_labels.head()
#Grouping the localities according to their Cluster Labels

blr_labels.sort_values(["Cluster Label"], inplace=True)

blr_labels.head()
#Plot the cluster on map

cluster_map = folium.Map(location=[blr_lat, blr_lng],zoom_start=11)



#marker for Bangalore

folium.Marker([blr_lat, blr_lng], popup='<i>Bangalore</i>', color='red', tooltip="Click to see").add_to(cluster_map)



#Getting the colors for the clusters

col = ['red', 'green', 'blue']



#markers for localities

for latitude,longitude,name,clus in zip(blr_labels["Latitudes"], blr_labels["Longitudes"], blr_labels["Locality"], blr_labels["Cluster Label"]):

    label = folium.Popup(name + ' - Cluster ' + str(clus))

    folium.CircleMarker(

        [latitude, longitude],

        radius=6,

        color=col[clus],

        popup=label,

        fill=False,

        fill_color=col[clus],

        fill_opacity=0.3

    ).add_to(cluster_map)

       

cluster_map
#First Cluster

cluster_1 = blr_labels[blr_labels['Cluster Label'] == 0]

print("There are {} localities in cluster-1".format(cluster_1.shape[0]))

mean_presence_1 = cluster_1['Italian Restaurant'].mean()

print("The mean occurence of Italian restaurant in cluster-1 is {0:.2f}".format(mean_presence_1))

cluster_1
#Second Cluster

cluster_2 = blr_labels[blr_labels['Cluster Label'] == 1]

print("There are {} localities in cluster-2".format(cluster_2.shape[0]))

mean_presence_2 = cluster_2['Italian Restaurant'].mean()

print("The mean occurence of Italian restaurant in cluster-2 is {0:.2f}".format(mean_presence_2))

cluster_2
#Third Cluster

cluster_3 = blr_labels[blr_labels['Cluster Label'] == 2]

print("There are {} localities in cluster-3".format(cluster_3.shape[0]))

mean_presence_3 = cluster_3['Italian Restaurant'].mean()

print("The mean occurence of Italian restaurant in cluster-3 is {0:.2f}".format(mean_presence_3))

cluster_3