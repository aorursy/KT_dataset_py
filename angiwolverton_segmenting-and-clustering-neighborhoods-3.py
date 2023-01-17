import numpy as np # library to handle data in a vectorized manner



import pandas as pd # library for data analsysis

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



import json # library to handle JSON files





#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab

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

from folium import plugins

from folium.plugins import MarkerCluster



print('Libraries imported.')
#read table using pandas, table[0] because there were several tables on the page and we only want the first

url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

table = pd.read_html(url)

table[0].head()
#replace 'Not assigned' with 'NaN'

table[0].replace("Not assigned", np.nan, inplace=True)

table[0].head()
#drop unassigned boroughs

table[0].dropna(subset=["Borough"], axis=0, inplace=True)

table[0].head(10)
#reset index

table[0].reset_index(drop=True, inplace=True)

table[0].head(10)
!pip install geocoder # This came as a recommendation. The original was !conda install -c conda-forge geocoder --yes
import geocoder

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
#get latitude and longitude using geocoder



# initialize your variable to None

lat_lng_coords = None



# loop until you get the coordinates

while(lat_lng_coords is None):

  g = geocoder.arcgis('{}, Toronto, Ontario'.format('Postal Code'))

  lat_lng_coords = g.latlng



latitude = lat_lng_coords[0]

longitude = lat_lng_coords[1]



print(latitude,longitude )
#read geospatial data file

url = 'http://cocl.us/Geospatial_data'

geotable = pd.read_csv(url)

geotable.head()
#append lat & long into table[0]

toronto = pd.merge(table[0], geotable, left_on='Postal Code', right_on='Postal Code', left_index=False, right_index=False)

toronto.head()
toronto.shape
print('The dataframe has {} boroughs'.format(len(toronto['Borough'].unique())))
#make a map of all Toronto neighborhoods

#Get the coordinates of Toronto

lat_lng_coords = None



# loop until you get the coordinates

while(lat_lng_coords is None):

  g = geocoder.arcgis('{}, Toronto, Ontario')

  lat_lng_coords = g.latlng



latitude = lat_lng_coords[0]

longitude = lat_lng_coords[1]



#create a folium map of Toronto with Boroughs

map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

for lat, lng, borough, neighborhood in zip(toronto['Latitude'], toronto['Longitude'], toronto['Borough'], toronto['Neighborhood']):

    label = '{}, {}'.format(neighborhood, borough)

    label = folium.Popup(label)

    folium.CircleMarker([lat, lng], radius=5, popup=label, color='blue', fill_color='blue').add_to(map_Toronto)

map_Toronto
#Narrow down our dataset to just Downtown Toronto since it looks a bit congested and we'll take a closer look.

downtown = toronto[toronto['Borough'] == 'Downtown Toronto'].reset_index(drop=True)

downtown.head()
downtown.shape
#make a map of our Downtown neighborhoods.

lat_lng_coords = None



# loop until you get the coordinates

while(lat_lng_coords is None):

  g = geocoder.arcgis('{}, Toronto, Ontario'.format('Postal_Code'))

  lat_lng_coords = g.latlng



latitude = lat_lng_coords[0]

longitude = lat_lng_coords[1]



map_Downtown = folium.Map(location=[latitude, longitude], zoom_start=13

                        )

for lat, lng, neighborhood in zip(downtown['Latitude'], downtown['Longitude'], downtown['Neighborhood']):

    label = '{}'.format(neighborhood)

    label = folium.Popup(label)

    folium.CircleMarker([lat, lng], radius=5, popup=label, color='blue', fill_color='blue').add_to(map_Downtown)

map_Downtown
# I started working on this one when my maps disappeared, so it isn't finished, nor is it likely to work. I'll come back to this later.

#Let's see if we can group those markers by Borough.



#mc = MarkerCluster()



#map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

#for lat, lng, borough, neighborhood in zip(toronto['Latitude'], toronto['Longitude'], toronto['Borough'], toronto['Neighborhood']):

#    label = '{}'.format(neighborhood, borough)

 #   label = folium.Popup(label)

  #  for row in toronto:

   #     mc.add_child(folium.Marker(location=['Borough'],

    #             popup=label))

    #map_Toronto.add_child(mc)

    #folium.CircleMarker([lat, lng], radius=5, popup=label, color='blue', fill_color='blue').add_to(map_Toronto)

#map_Toronto

# Let's put in the Foursquare credentials for the rest of this project.

CLIENT_ID = 'VFWNLRVPD2FA4JFQOWMKPIFBTBUBQUITNC3GKVDWY2FA2TRF'

CLIENT_SECRET = 'NQOIJKU4YIO1BPSS50QUOJNV2LZZPXKKD2X0LJVOEL5SNBKK'

VERSION = '20180605'

print('Your credentials:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET: ' + CLIENT_SECRET)
# Let's explore the first neighborhood in the dataframe. If I have time, I'd like to do this again with more recent data because of the Stay at Home orders that are out there.

RegentHarbour = downtown



neighborhood_name = RegentHarbour.loc[0, 'Neighborhood']

neighborhood_latitude = RegentHarbour.loc[0, 'Latitude']

neighborhood_longitude = RegentHarbour.loc[0, 'Longitude']

print('The latitude and longitude of {} are {} and {}.'.format(neighborhood_name, neighborhood_latitude, neighborhood_longitude))
# Get the top 100 venues in Regent Park and Harbourfront.

LIMIT = 100

radius = 500

Top100 = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, neighborhood_latitude, neighborhood_longitude, VERSION, radius, LIMIT)

Top100
results = requests.get(Top100).json()

results
# We're going to need a list of categories.

def get_category_type(row):

    try:

        categories_list = row['categories']

    except:

        categories_list = row['venue.categories']

            

    if len(categories_list) == 0:

        return None

    else:

        return categories_list[0]['name']
# Let's clean this up a bit and put it into a dataframe. For fun, I'm going to include the address in my df, not just category.

venues = results['response']['groups'][0]['items']

nearby_venues = json_normalize(venues) # flatten JSON



# Filter the columns that we'd like in the df.

filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng', 'venue.location.address']

nearby_venues = nearby_venues.loc[:, filtered_columns]



#Filter the category for each row. (I feel like this could be considered applying our category definition to the df, how we'd like it displayed.)

nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)



#Define how we'd like to separate the columns.

nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]



nearby_venues.head()
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
# We have all the Boroughs in Toronto still in a df.

toronto.head()
#We also have all the Neighborhoods in Downtown Toronto in a df.

downtown.head()
# Let's get that list of venues in a df with the lat/long for the venue.

def getNearbyVenues(names, latitudes, longitudes, radius=500):

    venues_list=[]

    for name, lat, lng, in zip(names, latitudes, longitudes):

        print(name)

        #api request

        url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, lat, lng, VERSION, radius, LIMIT)

        #get request

        results = requests.get(url).json()['response']['groups'][0]['items']

        

        #return only the info we want

        venues_list.append([(name, lat, lng,

                           v['venue']['name'],

                           v['venue']['location']['lat'],

                           v['venue']['location']['lng'],

                           v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

    nearby_venues.columns = ['Neighborhood'],['Neighborhood Latitude'],['Neighborhood Longitude'],['Venue'],['Venue Latitude'],['Venue Longitude'],['Venue Category']

            

    return(nearby_venues)
downtown_venues = getNearbyVenues(names=downtown['Neighborhood'], latitudes=downtown['Latitude'], longitudes=downtown['Longitude'])
print(downtown_venues.shape)

downtown_venues.head()
list(downtown_venues.columns.values)
# Haha! What happened to my column names?!

downtown_venues.columns = ['Neighborhood', 'Neighborhood Latitude', 'Neighborhood Longitude', 'Venue', 'Venue Latitude', 'Venue Longitude', 'Venue Category']

downtown_venues.head()
#How many venues are in each Neighborhood?

downtown_venues.groupby('Neighborhood').count()
#How many unique categories do we have?

print('There are {} unique categories.'.format(len(downtown_venues['Venue Category'].unique())))
#one hot encoding

downtown_onehot = pd.get_dummies(downtown_venues[['Venue Category']], prefix="", prefix_sep="")



#add neighborhood column

downtown_onehot['Neighborhood'] = downtown_venues['Neighborhood']



#move neighborhood to first column

fixed_columns = [downtown_onehot.columns[-1]] + list(downtown_onehot.columns[:-1])

downtown_onehot = downtown_onehot[fixed_columns]



print(downtown_onehot.shape)

downtown_onehot.head()
#group rows by neighborhood and find freqency of each category

downtown_grouped = downtown_onehot.groupby('Neighborhood').mean().reset_index()

downtown_grouped
downtown_grouped.shape
#Find the top 5 venues for each neighborhood.

num_top_venues = 5



for hood in downtown_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = downtown_grouped[downtown_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue', 'freq']

    temp = temp.iloc[1:]

    temp['freq'] = temp['freq'].astype(float)

    temp = temp.round({'freq': 2})

    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))

    print('\n')
#Let's put this in a df by the most common venues



#Start by putting our categories in order

def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_venues]
#Create the dataframe with the top 10 venues for each neighborhood.

num_top_venues = 10



indicators = ['st', 'nd', 'rd']  #As in 1st, 2nd, 3rd



#create the columns

columns = ['Neighborhood']

for ind in np.arange(num_top_venues):

    try:

        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind])) #column_number+'st', column_number+'nd', column_number+'rd'

    except:

        columns.append('{}th Most Common Venue'.format(ind+1)) #4th, 5th, etc

    

#create df

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)

neighborhoods_venues_sorted['Neighborhood'] = downtown_grouped['Neighborhood']



for ind in np.arange(downtown_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(downtown_grouped.iloc[ind, :], num_top_venues)

    

neighborhoods_venues_sorted.head()
#I'm going to try narrowing it down to just the Boroughs with Toronto in their names.

toronto_buroughs = toronto[toronto['Borough'].astype(str).str.contains('Toronto')]

toronto_buroughs
# Let's get that list of venues in a df with the lat/long for the venue.

latitudes = toronto_buroughs['Latitude']

longitudes = toronto_buroughs['Longitude']



def getNearbyVenues(names, latitudes, longitudes, radius=500):

    venues_list=[]

    for name, lat, lng, in zip(names, latitudes, longitudes):

        print(name)

        #api request

        url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, lat, lng, VERSION, radius, LIMIT)

        #get request

        results = requests.get(url).json()['response']['groups'][0]['items']

        

        #return only the info we want

        venues_list.append([(name, lat, lng,

                           v['venue']['name'],

                           v['venue']['location']['lat'],

                           v['venue']['location']['lng'],

                           v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

    nearby_venues.columns = ['Neighborhood'],['Neighborhood Latitude'],['Neighborhood Longitude'],['Venue'],['Venue Latitude'],['Venue Longitude'],['Venue Category']

            

    return(nearby_venues)
toronto_buroughs_venues = getNearbyVenues(names=toronto_buroughs['Neighborhood'], latitudes=toronto_buroughs['Latitude'], longitudes=toronto_buroughs['Longitude'])
print(toronto_buroughs_venues.shape)

toronto_buroughs_venues
# Stupid column names

toronto_buroughs_venues.columns = ['Neighborhood', 'Neighborhood Latitude', 'Neighborhood Longitude', 'Venue', 'Venue Latitude', 'Venue Longitude', 'Venue Category']

toronto_buroughs_venues.head()
#How many are in each neighborhood?

toronto_buroughs_venues.groupby('Neighborhood').count()
print('There are {} unique categories.'.format(len(toronto_buroughs_venues['Venue Category'].unique())))
#one hot

toronto_buroughs_onehot = pd.get_dummies(toronto_buroughs_venues[['Venue Category']], prefix="", prefix_sep="")



toronto_buroughs_onehot['Neighborhood'] = toronto_buroughs_venues['Neighborhood']



fixed_columns = [toronto_buroughs_onehot.columns[-1]] + list(toronto_buroughs_onehot.columns[:-1])

toronto_buroughs_onehot = toronto_buroughs_onehot[fixed_columns]



toronto_buroughs_onehot.head()
toronto_buroughs_onehot.shape
toronto_buroughs_grouped = toronto_buroughs_onehot.groupby('Neighborhood').mean().reset_index()

toronto_buroughs_grouped
toronto_buroughs_grouped.shape
#Top 5 venue from each neighborhood.

num_top_venues = 5



for hood in toronto_buroughs_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = toronto_buroughs_grouped[toronto_buroughs_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue', 'freq']

    temp = temp.iloc[1:]

    temp['freq'] = temp['freq'].astype(float)

    temp = temp.round({'freq': 2})

    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))

    print('\n')
#Let's recreate our Top 10 dataframe

def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_venues]
num_top_venues = 10



indicators = ['st', 'nd', 'rd']



columns = ['Neighborhood']

for ind in np.arange(num_top_venues):

    try:

        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common Venue'.format(ind+1))

        

Tburoughs_venues_sorted = pd.DataFrame(columns=columns)

Tburoughs_venues_sorted['Neighborhood'] = toronto_buroughs_grouped['Neighborhood']



for ind in np.arange(toronto_buroughs_grouped.shape[0]):

    Tburoughs_venues_sorted.iloc[ind,1:] = return_most_common_venues(toronto_buroughs_grouped.iloc[ind, :], num_top_venues)

    

Tburoughs_venues_sorted.head()
# Now for the good part!

#k-means clustering

kclusters = 5

tb_grouped_clustering = toronto_buroughs_grouped.drop('Neighborhood', 1)

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(tb_grouped_clustering)

kmeans.labels_[0:10]
#Create a new df that includes the cluster and Top 10 for each neighborhood

#add clustering labels

Tburoughs_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

TB_merged = toronto_buroughs



#merge TB_merged w/ toronto to add lat/long

TB_merged = TB_merged.join(Tburoughs_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



TB_merged.head()
#Let's put them all on a map!

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)



#set colors scheme for clusters

x = np.arange(kclusters)

ys = [i + x +(i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



#add markers

markers_colors = []

for lat, lon, poi, cluster in zip(TB_merged['Latitude'], TB_merged['Longitude'], TB_merged['Neighborhood'], TB_merged['Cluster Labels']):

    label = folium.Popup(str(poi) + 'Cluster' + str(cluster), parse_html=True)

    folium.CircleMarker([lat, lon],

                       radius=5, 

                       popup=label, 

                       color=rainbow[cluster-1], 

                       fill=True, 

                       fill_color=rainbow[cluster-1], 

                       fill_opacity=0.7).add_to(map_clusters)

    

map_clusters
#Let's take a look at the first cluster.

TB_merged.loc[TB_merged['Cluster Labels'] == 0, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#The second

TB_merged.loc[TB_merged['Cluster Labels'] == 1, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#The third

TB_merged.loc[TB_merged['Cluster Labels'] == 2, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#The fourth

TB_merged.loc[TB_merged['Cluster Labels'] == 3, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#The fifth

TB_merged.loc[TB_merged['Cluster Labels'] == 4, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
# determine k using elbow method



from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

import numpy as np

import matplotlib.pyplot as plt



# k means determine k

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(tb_grouped_clustering)

    kmeanModel.fit(tb_grouped_clustering)

    distortions.append(sum(np.min(cdist(tb_grouped_clustering, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / tb_grouped_clustering.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
kclusters = 3

tb_grouped_clustering = toronto_buroughs_grouped.drop('Neighborhood', 1)

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(tb_grouped_clustering)

kmeans.labels_[0:10]
#Create a new df that includes the cluster and Top 10 for each neighborhood

#add clustering labels

#Tburoughs_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

TB_merged = toronto_buroughs



#merge TB_merged w/ toronto to add lat/long

TB_merged = TB_merged.join(Tburoughs_venues_sorted.set_index('Neighborhood'), on='Neighborhood', how = 'right')



TB_merged.head()
kmeans.labels_[0:10]
#Let's put them all on a map!

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)



#set colors scheme for clusters

x = np.arange(kclusters)

ys = [i + x +(i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



#add markers

markers_colors = []



for cluster in range(0,kclusters): 

    group = folium.FeatureGroup(name='<span style=\\"color: {0};\\">{1}</span>'.format(rainbow[cluster-1],cluster))

    for lat, lon, poi, label in zip(TB_merged['Latitude'], TB_merged['Longitude'], TB_merged['Neighborhood'], TB_merged['Cluster Labels']):

        if int(label) == cluster: 

            label = folium.Popup('ORIG. '+ str(poi) + 'Cluster ' + str(cluster), parse_html=True)

            folium.CircleMarker(

                (lat, lon),

                radius=5,

                popup=label,

                color=rainbow[cluster-1],

                fill=True,

                fill_color=rainbow[cluster-1],

                fill_opacity=0.7).add_to(group)

    group.add_to(map_clusters)



    

map_clusters





#Let's take a look at the first cluster.

TB_merged.loc[TB_merged['Cluster Labels'] == 0, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#Let's take a look at the second cluster.

TB_merged.loc[TB_merged['Cluster Labels'] == 1, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#Let's take a look at the third cluster.

TB_merged.loc[TB_merged['Cluster Labels'] == 2, TB_merged.columns[[1] + list(range(5, TB_merged.shape[1]))]]
#Coffee Shops and Cafes seem to be pretty common in the first segment, but there still seem to be a few venues that don't quite seem to fit.