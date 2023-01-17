# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import urllib.request

from bs4 import BeautifulSoup

from geopy.geocoders import Nominatim

import requests # library to handle requests

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

import matplotlib.cm as cm# Matplotlib and associated plotting modules

import matplotlib.colors as colors

from sklearn.cluster import KMeans# import k-means from clustering stage

import folium # map rendering library

import os
# Wikipedia page containing th neighborhoods I'll be scraping

url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

# open the url using urllib.request and put the HTML into the page variable

page = urllib.request.urlopen(url)

# Beautiful soup import completed in the impor section

# parse the HTML from our URL into the BeautifulSoup parse tree format

soup = BeautifulSoup(page, "lxml")
#The table of interest has a 'wikitable' class defined. statement below pulls all objects with that class. Second statement display content found

neighborhood_table=soup.find('table', class_='wikitable')
#Add each column into an list. The resulting arrays will then be converted to a DataFrame

A=[]

B=[]

C=[]

for row in neighborhood_table.findAll('tr'):

    cells=row.findAll('td')

    if len(cells)==3:

        A.append(cells[0].find(text=True))

        B.append(cells[1].find(text=True))

        C.append(cells[2].find(text=True))

for n in range (0, len(C)):

    if C[n]=="\n":

        C[n]=B[n]
#Create a dataframe with the resulting arrays. Clear line breaks, remove not assigned Postal Codes

df = pd.DataFrame({'Postal Code':A,'Borough': B, 'Neighborhood':C})

df

df = df.replace(r'\n','', regex=True)

df = df.replace(r'/',',', regex=True)

df.drop(df.loc[df['Borough']=="Not assigned"].index, inplace=True)

df
df.shape
df2 = pd.read_csv('https://cocl.us/Geospatial_data')

df2
df3 = pd.merge(df,df2, on ="Postal Code")

df3
address = 'Toronto, CA'



geolocator = Nominatim(user_agent="to_explorer")

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
# create map of New York using latitude and longitude values

map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10, width=600, height= 500)

# add markers to map

for lat, lng, borough, neighborhood in zip(df3['Latitude'], df3['Longitude'], df3['Borough'], df3['Neighborhood']):

    label = '{}, {}'.format(neighborhood, borough)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_toronto)  

map_toronto
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

CLIENT_ID = user_secrets.get_secret("CLIENT_ID")

CLIENT_SECRET = user_secrets.get_secret("CLIENT_SECRET")

VERSION = '20180605' # Foursquare API version

LIMIT=100
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
toronto_venues = getNearbyVenues(names=df3['Neighborhood'],

                                   latitudes=df3['Latitude'],

                                   longitudes=df3['Longitude']

                                  )
toronto_venues.head()
print('Dataframe shape: ', toronto_venues.shape)
toronto_venues.groupby('Neighborhood').count()
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
# one hot encoding

toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 



# move neighborhood column to the first column

toronto_onehot = toronto_onehot[ ['Neighborhood'] + [ col for col in toronto_onehot.columns if col != 'Neighborhood' ] ]
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()

num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()

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

neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']



for ind in np.arange(toronto_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted
# set number of clusters

kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10] 
# add clustering labels

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df3

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.dropna(inplace=True)
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] ==3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10, width=600, height=500)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[int(cluster) -1],

        fill=True,

        fill_color=rainbow[int(cluster)-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters