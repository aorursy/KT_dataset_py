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

import json
import pandas as pd
import geojson
import geojsonio

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
with open('PTN.geojson', 'r') as f:
    network_data = json.load(f)

power_network=network_data['features']
power_network[0]
# define the dataframe columns
column_names = ['Node name', 'Description','Node type', 'Latitude', 'Longitude'] 

# instantiate the dataframe
PTN = pd.DataFrame(columns=column_names)
PTN

for data in power_network:
    name = data['properties']['name'] 
    description = data['properties']['other_tags']
        
    network_type = data['geometry']['type']
    network_latlon = data['geometry']['coordinates'][0]
    network_lat = network_latlon[1]
    network_lon = network_latlon[0]
    
    PTN = PTN.append({'Node name': name,
                                          'Description': description,
                                          'Node type':network_type,
                                          'Latitude': network_lat,
                                          'Longitude': network_lon}, ignore_index=True)
PTN.head()
PTN.groupby('Node type').count()
#As all datapoints are 'LineString' we can drop the entire column from our DF
PTN=PTN.drop('Node type', axis=1)
PTN
# Since we only required grid station locations which have a 'Node name', we are going to delete rows with no 'Node name' which 
#are mostly cables and powerlines

PTN=PTN.dropna(how='any',axis=0)
PTN=PTN.reset_index()
PTN=PTN.drop('index',axis=1)
PTN.shape
PTN=PTN.sort_values(by=['Node name']).reset_index().drop('index', axis=1)

PTN
#Lets get the coordinates of Pakistan using geolocator
address = 'Islamabad, PK'

geolocator = Nominatim(user_agent="pk_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Pakistan are {}, {}.'.format(latitude, longitude))
# Lets create map of Pakistan using latitude and longitude values and mark the locations of the Grid stations
map_PTN = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(PTN['Latitude'], PTN['Longitude'], PTN['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_PTN)  
    
map_PTN
CLIENT_ID = 'TQ0EQVSWEW1PZSVXLNC0CPJ1QEXZIYAAH1GGUVKU4IOWS4GP' # your Foursquare ID
CLIENT_SECRET = 'WOAR3RL51SFCKQZ0QZKQ2YVQIUHXXWLAABMAAOC5NZL4USGN' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
index=PTN[PTN['Node name']=='IESCO Grid Station'].index
index
Grid_Lat=PTN.loc[37,'Latitude']
Grid_Lng=PTN.loc[37,'Longitude']
print('Coordinates of IESCO Grid Station:',Grid_Lat,Grid_Lng)

LIMIT=100
radius=5000
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            Grid_Lat, 
            Grid_Lng, 
            radius, 
            LIMIT)
results = requests.get(url).json()
results
results['response']['groups'][0]['items']
venues = results['response']['groups'][0]['items']
nearby_venues = json_normalize(venues)
nearby_venues
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng','venue.location.distance']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues
def getNearbyVenues(names, latitudes, longitudes, radius=15000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
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
        results = requests.get(url).json()["response"]["groups"][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],
            v['venue']['location']['distance'],
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Node name', 
                  'Node Latitude', 
                  'Node Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude',
                  'Venue distance',
                  'Venue Category']
    
    return(nearby_venues)
# We will call this PTN_venues (PTN Stands for Pakistan Transmission Network)
PTN_venues = getNearbyVenues(names=PTN['Node name'],
                                   latitudes=PTN['Latitude'],
                                   longitudes=PTN['Longitude']                                
                                      )
PTN_venues
PTN_venues.to_csv (r'E:\Data of old laptop\M.Ali\USB Data\CVs\IBM DS\Capstone Project\PTN_venues.csv', index = False, header=True)
PTN_venues.shape
# Lets visualize the venue locations on the map 

map_PTN = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

# add markers to map
for lat, lng, label in zip(PTN['Latitude'], PTN['Longitude'], PTN['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_PTN) 
    
for lat, lng, label in zip(PTN_venues['Venue Latitude'], PTN_venues['Venue Longitude'], PTN_venues['Venue']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        parse_html=False).add_to(map_PTN) 
    
map_PTN


df_g=PTN_venues.groupby('Node name', as_index=False).count()
df_g
Rohri=PTN_venues[PTN_venues['Node name']=='Rohri New']
Rohri_mean=Rohri['Venue distance'].mean()
Rohri['mean distance']=Rohri_mean
Rohri
Islamabad=PTN_venues[PTN_venues['Node name']=='IESCO Grid Station']
Islamabad_mean=Islamabad['Venue distance'].mean()
Islamabad['mean distance']=Islamabad_mean
Islamabad
# Now lets calculate the mean venue distance for all the Grid Station Nodes
mean_distance=[]

for n in df_g['Node name']:    
    x=PTN_venues[PTN_venues['Node name']==n]
    y=x['Venue distance'].mean()
    
    mean_distance.append(y)

mean_distance
distance=pd.DataFrame(mean_distance)
PTN.sort_values(by=['Node name'], inplace=True, ascending=True)
PTN=PTN.reset_index()
distance
PTN['No of Venues'] = np.where(PTN['Node name'].isin(df_g['Node name']),'True', 'False')
PTN
df_T=PTN[PTN['No of Venues']=='True']
df_T=df_T.sort_values(by=['Node name'])
df_F=PTN[PTN['No of Venues']=='False']
df_T=df_T.reset_index().drop('index',axis=1)
df_T=df_T.drop([53,57,86])
df_T=df_T.reset_index().drop('index',axis=1)
df_T['No of Venues']=df_g['Venue']
df_T
df_f=df_T.append(df_F, ignore_index=True)
df_f=df_f.drop('level_0',axis=1)
df_f['No of Venues'][107:117]=0
df_f['Mean distance']=distance
df_f
distance.max()
df_f['Mean distance'][107:117]=14593
df_f=df_f.sort_values(by=['Node name']).reset_index().drop('index', axis=1)
df_f
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0.5,0.5,2,1])
ax.bar(df_f['No of Venues'],df_f['Mean distance'])
plt.show()
df_s=pd.read_csv(r"E:\Data of old laptop\M.Ali\USB Data\CVs\IBM DS\Capstone Project\DF_Solar1.csv")
df_f['PVOUT']=df_s['PVOUT']
df_f
plt.scatter(df_f['Latitude'], df_f['PVOUT'], alpha=0.7)
plt.xlabel('Latitude', fontsize=18)
plt.ylabel('PVOUT', fontsize=16)

plt.show()
plt.scatter(df_f['Latitude'], df_f['No of Venues'], alpha=0.5)
plt.xlabel('Latitude', fontsize=18)
plt.ylabel('Number of Venues', fontsize=16)

plt.show()
plt.scatter(df_f['Longitude'], df_f['No of Venues'], alpha=0.5)
plt.xlabel('Number of Venues', fontsize=18)
plt.ylabel('Mean distances', fontsize=16)

plt.show()
plt.scatter(df_f['Longitude'], df_f['PVOUT'], alpha=0.5)
plt.xlabel('PVOUT', fontsize=18)
plt.ylabel('Mean distances', fontsize=16)

plt.show()
df_final=df_f.drop(['level_0','Node name','Description','Latitude','Longitude'], axis=1)
df_final
from sklearn.preprocessing import StandardScaler
X = df_final.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_final)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)
df_final["Clus_km"] = labels
df_final.head(5)
import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:, 1], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Distance from Venues', fontsize=18)
plt.ylabel('PVOUT', fontsize=16)

plt.show()
df_final[df_final['Clus_km']==0].mean()
df_final[df_final['Clus_km']==1].mean()
df_final[df_final['Clus_km']==2].mean()
df_f["Clus_km"] = labels
Cluster_0=df_f[df_f["Clus_km"]==0]
Cluster_1=df_f[df_f["Clus_km"]==1]
Cluster_2=df_f[df_f["Clus_km"]==2]
map_0 = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(Cluster_0['Latitude'], Cluster_0['Longitude'], Cluster_0['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='purple',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_0)  
    
map_0
map_1 = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(Cluster_1['Latitude'], Cluster_1['Longitude'], Cluster_1['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='green',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_1)  
    
map_1
map_2 = folium.Map(location=[latitude,longitude], zoom_start=6)

# add markers to map
for lat, lng, label in zip(Cluster_2['Latitude'], Cluster_2['Longitude'], Cluster_2['Node name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='yellow',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        parse_html=False).add_to(map_2)  
    
map_2
