import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import numpy as np # library to handle data in a vectorized manner

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values



import requests # library to handle requests

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe



import matplotlib.cm as cm

import matplotlib.colors as colors

import matplotlib.pyplot as plt

import seaborn as sns



# import k-means from clustering stage

from sklearn.cluster import KMeans

import folium # map rendering library
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/dubai_neighborhood.csv")

df.rename(columns = {"Community": "Neighborhood"} , inplace = True)

df.head()
df["Pop_Per_Area"] = df["Population(2000)"]/ df["Area(km2)"]

df = df[df["Pop_Per_Area"]> 2000]

df.shape
df.info()
list_lat =[]

list_lon = []

# !pip install geopy

for address in df["Neighborhood"]:

#     address = com

    try:

        geolocator = Nominatim(user_agent="ny_explorer")

        location = geolocator.geocode(address)

        latitude = location.latitude

        longitude = location.longitude

        print('The geograpical coordinate of {} are {}, {}.'.format(address, latitude, longitude))

        list_lat.append(latitude)

        list_lon.append(longitude)

    except:

        print("connection error")
# I have add the lon, lat list in case geopy connection error as it is not stable on kaggle notebook

list_lat = [25.28594185, 24.7542271, 25.282575549999997, 25.26305655, 25.4792894, 25.2400505, 25.2333597, 25.244402800000003, 24.24352055, 25.26630785, 25.27667725, 25.2738924, 32.2513936, 25.224953149999997, 25.2586171, 25.3790294, 25.19889775, 25.277041699999998, 25.233175, 25.2720137, 33.5165586, 25.265992349999998, 25.1327524, 25.1505288]

list_lon = [55.32944354478134, 46.8218325, 55.32013863040134, 55.3205840389995, 55.522152, 55.27745853149375, 55.2920503, 55.30475541735386, 55.72212505, 55.324221785431625, 55.30976273508145, 55.32262973029905, 35.0635981, 55.39050657119131, 55.3202189, 55.43072299448985, 55.25704939243696, 55.33729994920512, 55.2773708, 55.4364278, 44.7999567, 55.31742845625631, 55.20585675122375, 55.2087708]
df.insert(5, column="Latitude", value= list_lat)

df.insert(6, column="Longitude", value= list_lon)



df.shape

latitude = 25.276987

longitude = 55.296249



map_dubai = folium.Map(location=[latitude, longitude], zoom_start=12)



# add markers to map

for lat, lng, Neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):

    label = '{}'.format(Neighborhood)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_dubai)  

map_dubai
CLIENT_ID = 'C1YNQDWJUZXSZPDVDFFFSALYZRTJXHIQ2KINRZWBT4PXXMAF' # your Foursquare ID

CLIENT_SECRET = 'DR4UJBXNUZMU0OJOVMVQE1GCNUEYUWTH430ZY4NCPVJEMMI3' # your Foursquare Secret

VERSION = '20190608' # Foursquare API version



print('Your credentails:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
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
def getNearbyVenues(names, latitudes, longitudes, radius=500):

    LIMIT = 100

    radius= 1000

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
dubai_venues = getNearbyVenues(names=df["Neighborhood"],

                                   latitudes=df['Latitude'],

                                   longitudes=df['Longitude']

                                  )
dubai_venues.head()
dubai_venues.groupby("Neighborhood").count().reset_index().shape
dubai_onehot = pd.get_dummies(dubai_venues[['Venue Category']], prefix="", prefix_sep="")

dubai_onehot[["Neighborhood", "Venue"]] = dubai_venues[["Neighborhood", "Venue"]]



fixed_columns = [dubai_onehot.columns[-1]] + list(dubai_onehot.columns[:-1])

dubai_onehot = dubai_onehot[fixed_columns]



fixed_columns = [dubai_onehot.columns[-1]] + list(dubai_onehot.columns[:-1])

dubai_onehot = dubai_onehot[fixed_columns]



dubai_onehot.head()
dubai_grouped = dubai_onehot.groupby("Neighborhood").mean().reset_index()

dubai_grouped.head()
num_top_venues = 2



for hood in dubai_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = dubai_grouped[dubai_grouped['Neighborhood'] == hood].T.reset_index()

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

neighborhoods_venues_sorted['Neighborhood'] = dubai_grouped['Neighborhood']



for ind in np.arange(dubai_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(dubai_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted.head()
neighborhoods_venues_sorted.describe(include="O")
# set number of clusters

kclusters = 5



dubai_grouped_clustering = dubai_grouped.drop('Neighborhood', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dubai_grouped_clustering)

clus_labels = kmeans.labels_

# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10] 
dubai_merged = pd.DataFrame(dubai_grouped["Neighborhood"])



dubai_merged.insert(0,'Cluster Labels',clus_labels)



dubai_merged = dubai_merged.join(neighborhoods_venues_sorted.set_index("Neighborhood"),on = "Neighborhood")



dubai_merged = dubai_merged.join(df.set_index("Neighborhood"), on ="Neighborhood")



# dubai_merged = pd.concat([dubai_merged, df[["Latitude", "Longitude"]]], axis=1)



dubai_merged.dropna(inplace = True)



dubai_merged["Cluster Labels"] = dubai_merged["Cluster Labels"].astype("int")



dubai_merged.shape

# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=13)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(dubai_merged['Latitude'], dubai_merged['Longitude'], dubai_merged['Neighborhood'], dubai_merged['Cluster Labels']):

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
Target_Business ='Hotel'



result = dubai_grouped.pivot(index='Neighborhood', columns=Target_Business, values=Target_Business)



plt.rcParams.update(plt.rcParamsDefault)



%matplotlib inline



sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=2 , color_codes=True, rc=None)

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15) # fontsize of the tick labels

plt.rcParams['figure.figsize'] = [15, 15]

#plt.rc('figure', titlesize=30) 

plt.rcParams.update({'font.size': 8})

plt.title('Distribution of Hotels in Dubai', fontdict = {'fontsize' : 20}, pad=30)

sns.heatmap(result, annot=True, cmap='viridis', linewidths = 0.01 , linecolor='grey', cbar=True, fmt="g")
Target_Business ='Restaurant'



result = dubai_grouped.pivot(index='Neighborhood', columns=Target_Business, values=Target_Business)



plt.rcParams.update(plt.rcParamsDefault)



%matplotlib inline



sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=2 , color_codes=True, rc=None)

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15) # fontsize of the tick labels

plt.rcParams['figure.figsize'] = [15, 15]

#plt.rc('figure', titlesize=30) 

plt.rcParams.update({'font.size': 8})

plt.title('Distribution of Restaurant in Dubai', fontdict = {'fontsize' : 20}, pad=30)

sns.heatmap(result, annot=True, cmap='viridis', linewidths = 0.01 , linecolor='grey', cbar=True, fmt="g")
Target_Business ='Café'



result = dubai_grouped.pivot(index='Neighborhood', columns=Target_Business, values=Target_Business)



plt.rcParams.update(plt.rcParamsDefault)



%matplotlib inline



sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=2 , color_codes=True, rc=None)

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15) # fontsize of the tick labels

plt.rcParams['figure.figsize'] = [15, 15]

#plt.rc('figure', titlesize=30) 

plt.rcParams.update({'font.size': 8})

plt.title('Distribution of Café in Dubai', fontdict = {'fontsize' : 20}, pad=30)

sns.heatmap(result, annot=True, cmap='viridis', linewidths = 0.01 , linecolor='grey', cbar=True, fmt="g")
Target_Business ='Population(2000)'



result = df.pivot(index='Neighborhood', columns=Target_Business, values=Target_Business)



plt.rcParams.update(plt.rcParamsDefault)



%matplotlib inline



sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=2 , color_codes=True, rc=None)

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15) # fontsize of the tick labels

plt.rcParams['figure.figsize'] = [15, 15]

#plt.rc('figure', titlesize=30) 

plt.rcParams.update({'font.size': 8})

plt.title('Distribution of Population in Dubai', fontdict = {'fontsize' : 20}, pad=30)

sns.heatmap(result, annot=True, cmap='viridis', linewidths = 0.01 , linecolor='grey', cbar=True, fmt="g")
df_cluster0 = dubai_merged[dubai_merged["Cluster Labels"] == 0]

df_cluster0
df_cluster1 = dubai_merged[dubai_merged["Cluster Labels"] == 1]

df_cluster1
df_cluster2 = dubai_merged[dubai_merged["Cluster Labels"] == 2]

df_cluster2
df_cluster3 = dubai_merged[dubai_merged["Cluster Labels"] == 3]

df_cluster3
df_cluster4 = dubai_merged[dubai_merged["Cluster Labels"] == 4]

df_cluster4