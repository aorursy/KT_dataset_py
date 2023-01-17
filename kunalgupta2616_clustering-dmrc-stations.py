import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

from geopy.geocoders import Nominatim 
# convert an address into latitude and longitude values

import requests # library to handle requests

# Matplotlib and associated plotting modules
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
address = 'Delhi, India'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Delhi are {}, {}.'.format(latitude, longitude))
map_delhi = folium.Map(location=[latitude, longitude], zoom_start=11)  
map_delhi.save(outfile= "test_map.html")
map_delhi
from bs4 import BeautifulSoup

wiki_url = 'https://en.wikipedia.org/wiki/List_of_Delhi_Metro_stations'
wiki_page = requests.get(wiki_url).text
wiki_doc = BeautifulSoup(wiki_page, 'lxml')

rows = wiki_doc.find('table', {'class': 'wikitable sortable'}).findAll('tr')


df = pd.DataFrame()
 


lst = []
form = '{ "name": "%s",\
          "details": {"line":"[%s]",\
                      "latitude":0.0,\
                      "longitude":0.0 }}'
Station=[]
Line = []
for row in rows[1:]:
    items = row.find_all('td')
    try:
        if len(items)==8:
            Station.append(items[0].find('a').contents[0])
            Line.append(items[2].find('a').find('span').find('b').contents[0])
            lst.append(form % (items[0].find('a').contents[0],
               items[2].find('a').find('span').find('b').contents[0]))
    
    except Exception as e:
        continue

string = '['+','.join(lst)+']'

data = json.loads(string)

f = open('metro.json', 'w+')
f.write(json.dumps(data, indent=4))
f.close()
print(len(Station))
print(len(Line))
df['Station']=Station
df['Line']=Line
df
Latitude = []
Longitude = []
for stat in Station:
    try:
        try:
            try:
                address = "{} metro station, Delhi, India".format(stat)
                geolocator = Nominatim(user_agent="ny_explorer")
                location = geolocator.geocode(address)
                lat = location.latitude
                long = location.longitude
            except Exception as e:
                address = "{}, Delhi, India".format(stat)
                geolocator = Nominatim(user_agent="ny_explorer")
                location = geolocator.geocode(address)
                lat = location.latitude
                long = location.longitude
        except Exception as e:
            address = "{}, India".format(stat)
            geolocator = Nominatim(user_agent="ny_explorer")
            location = geolocator.geocode(address)
            lat = location.latitude
            long = location.longitude
    except Exception as d:
        lat=None
        long=None
    Latitude.append(lat)
    Longitude.append(long)
df['Latitude'] = Latitude
df['Longitude'] = Longitude
df.head(20)
# df.to_csv('DELHI_METRO_DATA.csv',index=False)
df=pd.read_csv('DELHI_METRO_DATA.csv')
df
linetonum = {"Yellow Line": 1, "Red Line": 2,"Blue Line": 3,'Blue Line branch':3, "Pink Line": 4,"Magenta Line": 5, "Green Line": 6,'Green Line branch':6, "Violet Line": 7, "Orange Line": 8,"Grey Line": 9}
data = df.dropna(axis=0)
data
data.replace({"Line": linetonum},inplace=True)
data
colors_dict = {1:'#FFFF00', 2:'#FF0000',3:'#0000FF', 4:'#FFC0CB',5:'#FF00FF', 6:'#008000',7:'#EE82EE', 8:'#FFA500',9:'#808080'} 
address = 'Delhi, India'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Delhi are {}, {}.'.format(latitude, longitude))
map_delhi_metro = folium.Map(location=[latitude, longitude], zoom_start=10)

for line, station, lat,long in zip(data['Line'], data['Station'],data['Latitude'], data['Longitude']):
    folium.Circle(
        [lat,long],
        popup=station,
        radius=20,
        color=colors_dict[line]
    ).add_to(map_delhi_metro)
map_delhi_metro.save(outfile= "outlier_map.html")
map_delhi_metro
data.at[97,'Latitude'] = 28.656682
data.at[97,'Longitude'] = 77.236612
data

# Sort the rows of dataframe by column 'Line'
data_sort = data.sort_values(by ='Line' )
data_sort
data_sort.dtypes
map_delhi_metro = folium.Map(location=[latitude, longitude], zoom_start=10)
#add markers
for line, station, lat,long in zip(data_sort['Line'], data_sort['Station'],data_sort['Latitude'], data_sort['Longitude']):
    folium.Circle(
        [lat,long],
        popup=station,
        radius=30,
        fill=True,
        color=colors_dict[line]
    ).add_to(map_delhi_metro)   

map_delhi_metro
CLIENT_ID = 'Your Client ID'
CLIENT_SECRET = 'Your Client Secret'
VERSION = '20180605'
categories_url = 'https://api.foursquare.com/v2/venues/categories?client_id={}&client_secret={}&v={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION)
            
# make the GET request
results = requests.get(categories_url).json()
len(results['response']['categories'])
categories_list = []
# Let's print only the top-level categories and their IDs and also add them to categories_list

def print_categories(categories, level=0, max_level=0):    
    if level>max_level: return
    out = ''
    out += '-'*level
    for category in categories:
        print(out + category['name'] + ' (' + category['id'] + ')')
        print_categories(category['categories'], level+1, max_level)
        categories_list.append((category['name'], category['id']))
        
print_categories(results['response']['categories'], 0, 0)
def get_venues_count(lat,long, radius, categoryId):
    explore_url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&categoryId={}'.format(
                CLIENT_ID, 
                CLIENT_SECRET, 
                VERSION,
                lat,
                long,
                radius,
                categoryId)
    try:
        return requests.get(explore_url).json()['response']['totalResults']
    except Exception as e:
        return 0
data_sort.reset_index(inplace=True,drop=True)
data_sort
stations_venues_df = data_sort.copy()
for c in categories_list:
    stations_venues_df[c[0]] = 0

for i, row in stations_venues_df[stations_venues_df.index > 179].iterrows():
    print(i)
    for c in categories_list:        
        stations_venues_df.loc[i, c[0]] = get_venues_count(stations_venues_df.Latitude.iloc[i],stations_venues_df.Longitude.iloc[i], radius=1000,categoryId=c[1])
    stations_venues_df.to_csv('stations_venues.csv')
stations_venues = pd.read_csv('stations_venues.csv', index_col=0)
stations_venues
stations_venues.shape
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.xticks(rotation='vertical')
sns.boxplot

ax = sns.boxplot(data = stations_venues)
ax.set_ylabel('Count of venues', fontsize=25)
ax.set_xlabel('Venue category', fontsize=25)
ax.tick_params(labelsize=20)
plt.xticks(rotation=45, ha='right')

plt.show()
from sklearn.preprocessing import MinMaxScaler

X = stations_venues.values[:,4:]
cluster_dataset = MinMaxScaler().fit_transform(X)
cluster_df = pd.DataFrame(cluster_dataset)
cluster_df.columns = [c[0] for c in categories_list]
cluster_df.head()
plt.figure(figsize=(20, 10))
plt.xticks(rotation='vertical')
sns.boxplot

ax = sns.boxplot(data = cluster_df)
ax.set_ylabel('Count of venues', fontsize=25)
ax.set_xlabel('Venue category', fontsize=25)
ax.tick_params(labelsize=20)
plt.xticks(rotation=45, ha='right')

plt.show()
from sklearn.cluster import KMeans 

Sum_of_squared_distances = []
K = range(2,11)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(cluster_df)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
from sklearn.metrics import silhouette_score
sil = []

kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2,kmax+1):
  kmeans = KMeans(n_clusters = k).fit(cluster_df)
  labels = kmeans.labels_
  sil.append(silhouette_score(cluster_df, labels, metric = 'euclidean'))
K1=range(2,kmax+1)
plt.plot(K1, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette_score')
plt.title('silhouette_score Method For Optimal k')
plt.show()
kclusters = 4

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(cluster_df)

kmeans_labels = kmeans.labels_
kmeans_labels
(unique, counts) = np.unique(kmeans_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)
replace_labels = {0:3,1:1,2:2,3:0}
for i in range(len(kmeans_labels)):
    kmeans_labels[i] = replace_labels[kmeans_labels[i]]

stations_clusters_df = stations_venues.copy()
stations_clusters_df['Cluster'] = kmeans_labels
stations_clusters_minmax_df = cluster_df.copy()
stations_clusters_minmax_df['Cluster'] = kmeans_labels
stations_clusters_minmax_df['Station'] = stations_venues['Station']
stations_clusters_minmax_df['Latitude'] = stations_venues['Latitude']
stations_clusters_minmax_df['Longitude'] = stations_venues['Longitude']
import matplotlib.ticker as ticker

fig, axes = plt.subplots(1,kclusters, figsize=(20, 10), sharey=True)

axes[0].set_ylabel('Count of venues (relative)', fontsize=25)
#plt.set_xlabel('Venue category', fontsize='x-large')

for k in range(kclusters):
    #Set same y axis limits
    axes[k].set_ylim(0,1.1)
    axes[k].xaxis.set_label_position('top')
    axes[k].set_xlabel('Cluster ' + str(k), fontsize=25)
    axes[k].tick_params(labelsize=20)
    plt.sca(axes[k])
    plt.xticks(rotation='vertical')
    sns.boxplot(data = stations_clusters_minmax_df[stations_clusters_minmax_df['Cluster'] == k].drop('Cluster',1), ax=axes[k])

plt.show()
address = 'Delhi, India'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Delhi are {}, {}.'.format(latitude, longitude))

cluster_map_delhi = folium.Map(location=[28.6517178,77.2219388], zoom_start=10)


#adding markers
for i, station, lat,long, cluster in zip(stations_clusters_minmax_df.index,
                                         stations_clusters_minmax_df['Station'],
                                         stations_clusters_minmax_df['Latitude'],
                                         stations_clusters_minmax_df['Longitude'],
                                         stations_clusters_minmax_df['Cluster']):
        
    
    colors=['blue','green','orange','red']
    
    station_series = stations_clusters_minmax_df.iloc[i]
    top_categories_dict = {}
    for cat in categories_list:
        top_categories_dict[cat[0]] = station_series[cat[0]]
    top_categories = sorted(top_categories_dict.items(), key = lambda x: x[1], reverse=True)
    popup='<b>{}</b><br>Cluster {}<br>1. {} {}<br>2. {} {}<br>3. {} {}'.format(
        station,
        cluster,
        top_categories[0][0],
        "{0:.2f}".format(top_categories[0][1]),
        top_categories[1][0],
        "{0:.2f}".format(top_categories[1][1]),
        top_categories[2][0],
        "{0:.2f}".format(top_categories[2][1]))
    folium.CircleMarker(
        [lat,long],
        fill=True,
        fill_opacity=0.5,
        popup=folium.Popup(popup, max_width = 300),
        radius=5,
        color=colors[cluster]
    ).add_to(cluster_map_delhi)
cluster_map_delhi.save('Cluster_DMRC_Stations.html')
cluster_map_delhi
cluster_0= stations_clusters_minmax_df[stations_clusters_minmax_df['Cluster']==0]
cluster_0
cluster_1= stations_clusters_minmax_df[stations_clusters_minmax_df['Cluster']==1]
cluster_1
cluster_2= stations_clusters_minmax_df[stations_clusters_minmax_df['Cluster']==2]
cluster_2
cluster_3= stations_clusters_minmax_df[stations_clusters_minmax_df['Cluster']==3]
cluster_3

