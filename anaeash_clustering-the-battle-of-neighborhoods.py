# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot

import seaborn as sns

# Too see full dataframe...

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', None)



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

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# First We have to locate the file path and changed accordingly

import os

os.getcwd()

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Link To Extract

path='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

# Read File

df_wiki=pd.read_html(path)

#Check the type

type(df_wiki)

# Call the position where the table is stored

neighborhood=df_wiki[0]

# Rename the Columns

neighborhood.rename(columns={0:'Postcode', 1: 'Borough', 2: 'Neighborhood'}, inplace=True)

# Eliminate the first row

neighborhood=neighborhood.drop([0])

# Eliminate "Not assigned", categorical values from "Borough" Column

neighborhood=neighborhood[neighborhood.Borough !='Not assigned']

# Making DataFrame

neighborhood=pd.DataFrame(neighborhood)

# Merging rows with same Postcode

neighborhood.set_index(['Postcode','Borough'],inplace=True)

merge_result = neighborhood.groupby(level=['Postcode','Borough'], sort=False).agg( ','.join)

# Setting the index

serial_wise=merge_result.reset_index()

# Assign the 'Borough' column value to 'Neighborhood' where 'Not assigned' occurs

serial_wise.loc[4, 'Neighborhood']='Queen\'s Park'

# Saving the file for future use!

serial_wise.to_excel('wikipedia_table.xls')

# Showing the Data Frame

df=pd.DataFrame(serial_wise)

df.head()
# Geographical Coordinates

df1=pd.read_csv("../input/geospatial-coordinates-toronto/Geospatial_Coordinates.csv")

# Change the Postal Code to Postcode

df1.rename(columns={'Postal Code':'Postcode'},inplace=True)

#Cancatenation

frames=[df,df1]

frames=pd.concat(frames, axis=1, sort=False)

# Merging the two columns on 'Postcode'

merge_columns=pd.merge(df, df1, left_on='Postcode', right_on='Postcode')

# Save the Data Frame

merge_columns.to_csv('neigbors_geographical.csv')

merge_columns.head()

# Sorting

# set index for only Downtown Toronto

downtown_toronto_data = merge_columns[merge_columns['Borough'] == 'Downtown Toronto'].reset_index(drop=True)

# eliminate 'Postcode' column

downtown_toronto_data=downtown_toronto_data.drop(['Postcode'], axis=1)

downtown_toronto_data.head()
neighborhoods=pd.read_csv("../input/neighborhoods-ny/neighborhoods_NY.csv", index_col=0)

# And make sure that the dataset has all 5 boroughs and 306 neighborhoods.

print('The dataframe has {} boroughs and {} neighborhoods.'.format(

        len(neighborhoods['Borough'].unique()),

        neighborhoods.shape[0]

    )

)



neighborhoods.head()
# Creating new Dataframe manhattan_data

manhattan_data = neighborhoods[neighborhoods['Borough'] == 'Manhattan'].reset_index(drop=True)

manhattan_data.head()
# Define Foursquare Credentials and Version

CLIENT_ID = 'HRMBKZUASN1NWO005IQK4TGG15UVEY5GCLJCYXHXW0VDP00K' # your Foursquare ID

CLIENT_SECRET = 'JSXFO23NR2OMICQSZRFQYDAZG1GMNRALXXACAFVNF5CGAM4C' # your Foursquare Secret

VERSION = '20180604'

limit = 20

print('Your credentails:')

print('CLIENT_ID:'+ CLIENT_ID)

print('CLIENT_SECRET:'+ CLIENT_SECRET)
# get the geographical coordinates of Downtown Toronto

address = 'Downtown Toronto, ON, Canada'



geolocator = Nominatim()

location = geolocator.geocode(address)

latitude_downtown_toronto = location.latitude

longitude_downtown_toronto = location.longitude

print("Downtown Toronto","latitude",latitude_downtown_toronto, "& " "longitude" ,longitude_downtown_toronto)
# Let's get the geographical coordinates of Manhattan.

address = 'Manhattan, NY'



geolocator = Nominatim()

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))
# create map of Downtown Toronto using latitude and longitude values

map_downtown_toronto = folium.Map(location=[latitude_downtown_toronto,longitude_downtown_toronto], zoom_start=11)



# add markers to map

for lat, lng, label in zip(downtown_toronto_data['Latitude'], downtown_toronto_data['Longitude'], downtown_toronto_data['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_downtown_toronto)  

    

map_downtown_toronto
from folium import plugins

# create map of Downtown Toronto using latitude and longitude values

map_downtown_toronto = folium.Map(location=[latitude_downtown_toronto,longitude_downtown_toronto], zoom_start=11)

# instantiate a mark cluster object for the incidents in the dataframe

incidents = plugins.MarkerCluster().add_to(map_downtown_toronto)

# add markers to map

for lat, lng, label in zip(downtown_toronto_data['Latitude'], downtown_toronto_data['Longitude'], downtown_toronto_data['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(incidents)  

    

map_downtown_toronto
# let's visualizat Manhattan the neighborhoods in it.

# create map of Manhattan using latitude and longitude values

map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to map

for lat, lng, label in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_manhattan)  

    

map_manhattan
# create map of Manhattan using latitude and longitude values

map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)



grouping = plugins.MarkerCluster().add_to(map_manhattan)



# add markers to map

for lat, lng, label in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(grouping)  

    

map_manhattan
# Let's create a function to repeat the process to all the neighborhoods in Toronto

def getNearbyVenues(names, latitudes,longitudes, radius=500):

    

    venues_list=[]

    for name, lat, lng in zip(names,latitudes,longitudes):

        print(name)

            

        # create the API request URL

        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(

            CLIENT_ID, 

            CLIENT_SECRET, 

            VERSION, 

            lat, 

            lng, 

            radius, 

            limit)

            

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
# Write the code to run the above function on each neighborhood and create a new dataframe called toronto_venues.

downtown_toronto_venues = getNearbyVenues(names=downtown_toronto_data['Neighborhood'],

                                   latitudes=downtown_toronto_data['Latitude'],

                                   longitudes=downtown_toronto_data['Longitude'],

                                  )
# Let's check the size of the resulting dataframe

print(downtown_toronto_venues.shape)

downtown_toronto_venues.head()
# Let's check how many venues were returned for each neighborhood

downtown_toronto_venues.groupby('Neighborhood').count()
# Let's find out how many unique categories can be curated from all the returned venues

print('There are {} uniques categories.'.format(len(downtown_toronto_venues['Venue Category'].unique())))
# one hot encoding

downtown_toronto_onehot = pd.get_dummies(downtown_toronto_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

downtown_toronto_onehot['Neighborhood'] = downtown_toronto_venues['Neighborhood'] 



# move neighborhood column to the first column

fixed_columns = [downtown_toronto_onehot.columns[-1]] + list(downtown_toronto_onehot.columns[:-1])

downtown_toronto_onehot = downtown_toronto_onehot[fixed_columns]



downtown_toronto_onehot.head()
# Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

downtown_toronto_grouped = downtown_toronto_onehot.groupby('Neighborhood').mean().reset_index()
# Let's print each neighborhood along with the top 5 most common venues

num_top_venues = 5



for hood in downtown_toronto_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = downtown_toronto_grouped[downtown_toronto_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue','freq']

    temp = temp.iloc[1:]

    temp['freq'] = temp['freq'].astype(float)

    temp = temp.round({'freq': 2})

    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))

    print('\n')
# Let's put that into a pandas dataframe

def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_venues]
# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

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

neighborhoods_venues_sorted['Neighborhood'] = downtown_toronto_grouped['Neighborhood']



for ind in np.arange(downtown_toronto_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(downtown_toronto_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted
# set number of clusters

kclusters = 5



downtown_toronto_grouped_clustering = downtown_toronto_grouped.drop('Neighborhood', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(downtown_toronto_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10] 
# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

downtown_toronto_merged = downtown_toronto_data



# add clustering labels

downtown_toronto_merged['Cluster Labels'] = kmeans.labels_



# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

downtown_toronto_merged = downtown_toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



downtown_toronto_merged.head() # check the last columns!
# create map

map_clusters = folium.Map(location=[latitude_downtown_toronto, longitude_downtown_toronto], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i+x+(i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(downtown_toronto_merged['Latitude'], downtown_toronto_merged['Longitude'], downtown_toronto_merged['Neighborhood'], downtown_toronto_merged['Cluster Labels']):

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
downtown_toronto_merged.loc[downtown_toronto_merged['Cluster Labels'] == 0, downtown_toronto_merged.columns[[1] + list(range(5, downtown_toronto_merged.shape[1]))]]
downtown_toronto_merged.loc[downtown_toronto_merged['Cluster Labels'] == 1, downtown_toronto_merged.columns[[1] + list(range(5, downtown_toronto_merged.shape[1]))]]
downtown_toronto_merged.loc[downtown_toronto_merged['Cluster Labels'] == 2, downtown_toronto_merged.columns[[1] + list(range(5, downtown_toronto_merged.shape[1]))]]
downtown_toronto_merged.loc[downtown_toronto_merged['Cluster Labels'] == 3, downtown_toronto_merged.columns[[1] + list(range(5, downtown_toronto_merged.shape[1]))]]
downtown_toronto_merged.loc[downtown_toronto_merged['Cluster Labels'] == 4, downtown_toronto_merged.columns[[1] + list(range(5, downtown_toronto_merged.shape[1]))]]
# Let's create a function to repeat the same process to all the neighborhoods in Manhattan

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

            limit)

            

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
# Now write the code to run the above function on each neighborhood and create a new dataframe called manhattan_venues

manhattan_venues = getNearbyVenues(names=manhattan_data['Neighborhood'],

                                   latitudes=manhattan_data['Latitude'],

                                   longitudes=manhattan_data['Longitude'],

                                  )
# Let's check how many venues were returned for each neighborhood

manhattan_venues.groupby('Neighborhood').count()
# Let's find out how many unique categories can be curated from all the returned venues

print('There are {} uniques categories.'.format(len(manhattan_venues['Venue Category'].unique())))
# one hot encoding

manhattan_onehot = pd.get_dummies(manhattan_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

manhattan_onehot['Neighborhood'] = manhattan_venues['Neighborhood'] 



# move neighborhood column to the first column

fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])

manhattan_onehot = manhattan_onehot[fixed_columns]



manhattan_onehot.head()
# Set Index

manhattan_grouped = manhattan_onehot.groupby('Neighborhood').mean().reset_index()
# Let's print each neighborhood along with the top 5 most common venues

num_top_venues = 5



for hood in manhattan_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = manhattan_grouped[manhattan_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue','freq']

    temp = temp.iloc[1:]

    temp['freq'] = temp['freq'].astype(float)

    temp = temp.round({'freq': 2})

    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))

    print('\n')
# Let's put that into a pandas dataframe

def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_venues]
# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

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

neighborhoods_venues_sorted['Neighborhood'] = manhattan_grouped['Neighborhood']



for ind in np.arange(manhattan_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(manhattan_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted
# Run k-means to cluster the neighborhood into 5 clusters.

# set number of clusters

kclusters = 5



manhattan_grouped_clustering = manhattan_grouped.drop('Neighborhood', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(manhattan_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10] 
# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood

manhattan_merged = manhattan_data



# add clustering labels

manhattan_merged['Cluster Labels'] = kmeans.labels_



# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

manhattan_merged = manhattan_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



manhattan_merged.head() # check the last columns!
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i+x+(i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(manhattan_merged['Latitude'], manhattan_merged['Longitude'], manhattan_merged['Neighborhood'], manhattan_merged['Cluster Labels']):

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
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 0, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 1, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 2, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 3, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 4, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]