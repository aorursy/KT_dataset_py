# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

!pip install geocoder

import geocoder

from geopy.geocoders import Nominatim

import folium

from pandas.io.json import json_normalize

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as colors

import seaborn as sns

from sklearn.cluster import KMeans

from pandas.plotting import table



print("All Libraries Loaded")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tehsil = pd.read_excel('/kaggle/input/Tehsil Data of Delhi.xlsx')

tehsil.columns = ['Neighborhood', 'Borough', 'Population', 'Literacy', 'Sex Ratio', 'Postal Code'] #Neighborhood: Tehsil of Delhi, Borough: District of Delhi in which the Tehsil is situated, Postal Code: NIC Tehsil Code

tehsil.head()
tehsil.shape
tehsil['Latitude'] = ""

tehsil['Longitude'] = ""

tehsil.head()
query = '{}, Chanakyapuri, New Delhi'

g = geocoder.arcgis(query.format(2085))

g.latlng
for i in range(tehsil.shape[0]):

    code = tehsil.loc[i, 'Postal Code']

    address = tehsil.loc[i, 'Neighborhood'] + ', ' + tehsil.loc[i, 'Borough']

    query = '{}, ' + address

    g = geocoder.arcgis(query.format(code))

    tehsil.loc[i, 'Latitude'] = g.latlng[0]

    tehsil.loc[i, 'Longitude'] = g.latlng[1]

tehsil.head()
tehsil.Literacy = tehsil.Literacy * 100
# Longitude and Latitude of Delhi

latitude = 28.7041

longitude = 77.1025



# create map of Delhi using latitude and longitude values

map_on = folium.Map(location=[latitude, longitude], zoom_start=10)



# add markers to map

for lat, lng, borough, neighborhood in zip(tehsil['Latitude'], tehsil['Longitude'], tehsil['Borough'], tehsil['Neighborhood']):

    label = '{}, {}'.format(neighborhood, borough)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=10,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_on)  

    

map_on
ratio = tehsil.groupby('Borough')['Sex Ratio'].mean()

pop = tehsil.groupby('Borough')['Population'].mean()

literacy = tehsil.groupby('Borough')['Literacy'].mean()



plt.style.use('ggplot')

fig = plt.figure(figsize = (20, 20))

ax_0 = fig.add_subplot(3, 1, 1) # add subplot 1 

ax_1 = fig.add_subplot(3, 1, 2) # add subplot 2

ax_2 = fig.add_subplot(3, 1, 3) # add subplot 2



#Subplot 1: Population

pop.plot.bar(sharex = True, ax = ax_0, color=['steelblue', 'steelblue', 'steelblue', 'steelblue', 'steelblue', 'crimson', 'steelblue', 'steelblue', 'rebeccapurple'])

ax_0.set_ylabel('Population', fontsize = 23)

ax_0.set_xlabel('Districts of Delhi', fontsize = 23)

ax_0.tick_params(axis='both', which='major', labelsize=20)

ax_0.set_title("Average Population in Each District of Delhi", fontdict = {'fontsize' : 20}, y = 1.04)



#Subplot 2: Sex Ratio

ratio.plot.bar(sharex = True, ax = ax_1, color=['gray', 'gray', 'lightseagreen', 'lightseagreen', 'gray', 'lightseagreen', 'lightseagreen', 'lightseagreen', 'rebeccapurple'])

ax_1.set_ylabel('Sex Ratio', fontsize = 23)

ax_1.set_xlabel('Districts of Delhi', fontsize = 23)

ax_1.tick_params(axis='both', which='major', labelsize=20)

ax_1.set_title("Average Sex Ratio in Each District of Delhi", fontdict = {'fontsize' : 20}, y = 1.04)



#Subplot 3: Literacy

literacy.plot.bar(sharex = True, ax = ax_2, color=['goldenrod', 'goldenrod', 'forestgreen', 'goldenrod', 'goldenrod', 'goldenrod', 'goldenrod', 'goldenrod', 'rebeccapurple'])

ax_2.set_ylabel('Literacy', fontsize = 23)

ax_2.set_xlabel('Districts of Delhi', fontsize = 23)

ax_2.tick_params(axis='both', which='major', labelsize=20)

ax_2.set_title("Average Literacy in Each District of Delhi", fontdict = {'fontsize' : 20}, y = 1.04)



fig.tight_layout(pad = 3.0)

plt.savefig('Visualizations by Districts of Delhi.png')

plt.show()
plt.style.available
ratio = tehsil.groupby('Neighborhood')['Sex Ratio'].mean().sort_values(ascending = False).head(7)

pop = tehsil.groupby('Neighborhood')['Population'].mean().sort_values(ascending = False).head(7)

literacy = tehsil.groupby('Neighborhood')['Literacy'].mean().sort_values(ascending = False).head(7)



plt.style.use('_classic_test_patch')

fig = plt.figure(figsize = (20, 30))

ax_0 = fig.add_subplot(3, 1, 1) # add subplot 1 

ax_1 = fig.add_subplot(3, 1, 2) # add subplot 2

ax_2 = fig.add_subplot(3, 1, 3) # add subplot 2



#Subplot 1: Population

pop.plot.bar(ax = ax_0, color='crimson')

ax_0.set_ylabel('Population', fontsize = 23)

ax_0.set_xlabel('Tehsil of Delhi', fontsize = 23)

ax_0.tick_params(axis='both', which='major', labelsize=20)

ax_0.set_title("Average Population in Each Tehsil of Delhi", fontdict = {'fontsize' : 20}, y = 1.04)



#Subplot 2: Sex Ratio

ratio.plot.bar(ax = ax_1, color='lightseagreen')

ax_1.set_ylabel('Sex Ratio', fontsize = 23)

ax_1.set_xlabel('Tehsil of Delhi', fontsize = 23)

ax_1.tick_params(axis='both', which='major', labelsize=20)

ax_1.set_title("Average Sex Ratio in Each Tehsil of Delhi", fontdict = {'fontsize' : 20}, y = 1.04)



#Subplot 3: Literacy

literacy.plot.bar(ax = ax_2, color='steelblue')

ax_2.set_ylabel('Literacy', fontsize = 23)

ax_2.set_xlabel('Tehsil of Delhi', fontsize = 23)

ax_2.tick_params(axis='both', which='major', labelsize=20)

ax_2.set_title("Average Literacy in Each Tehsil of Delhi", fontdict = {'fontsize' : 20}, y = 1.04)



fig.tight_layout(pad = 3.0)

plt.savefig('Visualizations by Tehsils of Delhi.png')

plt.show()
tehsil.head()
CLIENT_ID = 'SMW0KBR1QLGVOEBX3HFOQFVCABUB5HXWFTUZZCZMHFD5PVTW' 

CLIENT_SECRET = '3KAMZFVCZ15XQ04SVTKVDMVTEFQJASZNQRSVTUUNUZHX4PDZ'

VERSION = '20200406'



print('Your credentails:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
neighborhood_latitude = tehsil.loc[1, 'Latitude'] # Tehsil latitude value

neighborhood_longitude = tehsil.loc[1, 'Longitude'] # Tehsil longitude value



neighborhood_name = tehsil.loc[1, 'Neighborhood']# Tehsil name



print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 

                                                               neighborhood_latitude, 

                                                               neighborhood_longitude))
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(

    CLIENT_ID, 

    CLIENT_SECRET, 

    VERSION, 

    neighborhood_latitude, 

    neighborhood_longitude, 

    1020, 

    100)

url
results = requests.get(url).json()



# Defining function to extract categories

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
# The following function fetches venues from foursquare api:

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

            1020, 

            100)

            

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
tehsil_venues = getNearbyVenues(names=tehsil['Neighborhood'],

                                   latitudes=tehsil['Latitude'],

                                   longitudes=tehsil['Longitude']

                                  )
print(tehsil_venues.shape)

tehsil_venues.head(10)
df = pd.DataFrame(tehsil_venues.groupby('Neighborhood').count())

df
print('There are {} uniques categories.'.format(len(tehsil_venues['Venue Category'].unique())))
# one hot encoding

tehsil_onehot = pd.get_dummies(tehsil_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

tehsil_onehot['Neighborhood'] = tehsil_venues['Neighborhood'] 

tehsil_onehot.head()
tehsil_onehot.shape
tehsil_grouped = tehsil_onehot.groupby('Neighborhood').mean().reset_index()

tehsil_grouped
tehsil_grouped.shape
num_top_venues = 10



for hood in tehsil_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = tehsil_grouped[tehsil_grouped['Neighborhood'] == hood].T.reset_index()

    temp.columns = ['venue','freq']

    temp = temp.iloc[1:]

    temp['freq'] = temp['freq'].astype(float)

    temp = temp.round({'freq': 2})

    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))

    print('\n')
# Function to sort the venues in descending order

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

tehsil_venues_sorted = pd.DataFrame(columns=columns)

tehsil_venues_sorted['Neighborhood'] = tehsil_grouped['Neighborhood']



for ind in np.arange(tehsil_grouped.shape[0]):

    tehsil_venues_sorted.iloc[ind, 1:] = return_most_common_venues(tehsil_grouped.iloc[ind, :], num_top_venues)
df = tehsil_venues_sorted.head(10)

df
# set number of clusters

kclusters = 8



tehsil_grouped_clustering = tehsil_grouped.drop('Neighborhood', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(tehsil_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10]
# add clustering labels

tehsil_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)



tehsil_merged = tehsil



# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

tehsil_merged = tehsil_merged.join(tehsil_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



tehsil_merged.head() # check the last columns!
tehsil_merged.dropna(inplace=True)
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(tehsil_merged['Latitude'], tehsil_merged['Longitude'], tehsil_merged['Neighborhood'], tehsil_merged['Cluster Labels']):

    cluster = int(cluster)

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=25,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters
cluster1 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 0, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster1
cluster2 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 1, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster2
cluster3 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 2, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster3
cluster4 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 3, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster4
cluster5 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 4, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster5
cluster6 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 5, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster6
cluster7 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 6, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster7
cluster8 = tehsil_merged.loc[tehsil_merged['Cluster Labels'] == 7, tehsil_merged.columns[[1] + list(range(5, tehsil_merged.shape[1]))]]

cluster8
import seaborn as sns



fig = plt.figure(figsize=(65,65))

sns.set(font_scale=2.6)



ax = plt.subplot(3,1,1)

sns.violinplot(x="Neighborhood", y="Indian Restaurant", data=tehsil_onehot, cut=0);

plt.xlabel("")

plt.ylabel("Indian Restaurant", fontsize=50)

plt.xticks(rotation=45)



ax = plt.subplot(3,1,2)

sns.violinplot(x="Neighborhood", y="Café", data=tehsil_onehot, cut=0);

plt.xlabel("")

plt.ylabel("Cafe", fontsize=50)

plt.xticks(rotation=45)



plt.subplot(3,1,3)

sns.violinplot(x="Neighborhood", y="Pub", data=tehsil_onehot, cut=0);

plt.ylabel("Pub", fontsize=50)

plt.xticks(rotation=45)

plt.xlabel("Tehsils of Delhi", fontsize=50)

ax.text(-1.0, 3.1, 'Frequency distribution for the venue categories for each neighborhood', fontsize=60)

fig.tight_layout()

plt.savefig ("Distribution_Frequency_Venues_3_categories.png", dpi=300)

plt.show()
tehsil_select = tehsil_merged.loc[tehsil_merged.Neighborhood.isin(['Chanakya Puri', 'Connaught Place', 'Parliament Street', 'Vasant Vihar', 'Rajouri Garden'])]

tehsil_select
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(tehsil_select['Latitude'], tehsil_select['Longitude'], tehsil_select['Neighborhood'], tehsil_select['Cluster Labels']):

    cluster = int(cluster)

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=25,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters