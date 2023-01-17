!conda install -c conda-forge folium=0.5.0 --yes # comment/uncomment if not yet installed.
!conda install -c conda-forge geopy --yes        # comment/uncomment if not yet installed

import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis

# Numpy and Pandas libraries were already imported at the beginning of this notebook.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
# import k-means from clustering stage
from sklearn.cluster import KMeans
import folium # map rendering library

import requests # library to handle requests
import lxml.html as lh
import bs4 as bs
import urllib.request

print('Libraries imported.')
data_area=pd.read_csv("https://raw.githubusercontent.com/shayy07/Coursera_Capstone/master/OfficeArea.csv")
data_area
MyLM=data_area["Landmark"]
MyLM
# confidential
google_key="AIzaSyBkPw7COxbYTtvc4gTXDMRyxbYdxHHSo3M"
cid="R0A4V12LCTHAPQXIP5103KSSIWCCLOLSTZB3PRJUGMSPJNRN"
csecret="20SFFWC52FAZJEMZBRE450QB1H05F4YJL5QRGTK0I0F4ANRR"
data_area['Latitude'] = 0.0
data_area['Longitude'] = 0.0

for idx,area in data_area['Landmark'].iteritems():
    area=area + ' ' + 'Singapore'
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(area,google_key)
    #url='https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={}&key={}.format(area,google_key)
    lat = requests.get(url).json()["results"][0]["geometry"]["location"]['lat']
    lng = requests.get(url).json()["results"][0]["geometry"]["location"]['lng']
    data_area.loc[idx,'Latitude'] = lat
    data_area.loc[idx,'Longitude'] = lng
geo = Nominatim(user_agent='My-IBMNotebook')
address = 'Singapore'
location = geo.geocode(address)
latitude = location.latitude
longitude = location.longitude

# create map
map_area = folium.Map(location=[latitude, longitude], tiles="Openstreetmap", zoom_start=11)

# set color scheme for the clusters
x = np.arange(7)
ys = [i+x+(i*x)**2 for i in range(7)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
cluster=0
for lat, lon, poi, cat in zip(data_area['Latitude'], data_area['Longitude'], data_area['Area'], data_area['Landmark']):
    cluster=cluster+1
    label = folium.Popup(str(cat) + '-' + str(poi), parse_html=True)
    folium.Marker(
        [lat, lon],
        popup=label).add_to(map_area)
       
map_area
url = 'https://api.foursquare.com/v2/venues/explore'
venue_list=[]
for idx,lat in data_area['Latitude'].iteritems():
    lng=data_area.loc[idx,'Longitude']
    name=data_area.loc[idx,'Area']
    sll=str(lat) + ',' + str(lng)
    params = dict(
      client_id=cid,
      client_secret=csecret,
      v='20180323',
      ll=sll,
      radius=300,
      limit=80,
      query='food'
    )
    resp = requests.get(url=url, params=params).json()["response"]['groups'][0]['items']
    main_category="Other Food"
    venue_list.append([(name,lat,lng,main_category,v['venue']['id'],v['venue']['name'],v['venue']['location']['lat'],v['venue']['location']['lng'],v['venue']['categories'][0]['name']) for v in resp])
for idx,lat in data_area['Latitude'].iteritems():
    lng=data_area.loc[idx,'Longitude']
    name=data_area.loc[idx,'Area']
    sll=str(lat) + ',' + str(lng)
    params = dict(
      client_id=cid,
      client_secret=csecret,
      v='20180323',
      ll=sll,
      radius=300,
      limit=80,
      query='Café'
    )
    resp = requests.get(url=url, params=params).json()["response"]['groups'][0]['items']
    main_category="Café"
    venue_list.append([(name,lat,lng,main_category,v['venue']['id'],v['venue']['name'],v['venue']['location']['lat'],v['venue']['location']['lng'],v['venue']['categories'][0]['name']) for v in resp])
for idx,lat in data_area['Latitude'].iteritems():
    lng=data_area.loc[idx,'Longitude']
    name=data_area.loc[idx,'Area']
    sll=str(lat) + ',' + str(lng)
    params = dict(
      client_id=cid,
      client_secret=csecret,
      v='20180323',
      ll=sll,
      radius=300,
      limit=80,
      query='gym'
    )
    resp = requests.get(url=url, params=params).json()["response"]['groups'][0]['items']
    main_category="Gym"
    venue_list.append([(name,lat,lng,main_category,v['venue']['id'],v['venue']['name'],v['venue']['location']['lat'],v['venue']['location']['lng'],v['venue']['categories'][0]['name']) for v in resp])
nearby_venues = pd.DataFrame([item for venue_list in venue_list for item in venue_list])
nearby_venues.columns = ['Area','Area_Latitude','Area_Longitude','Venue_Main_Category','Venue_ID','Venue','Venue_Latitude','Venue_Longitude','Venue_Category']

nearby_venues.head()
nearby_venues.groupby(['Venue_Main_Category'])['Venue_Category'].value_counts(normalize=False)
nearby_venues_clean=nearby_venues.copy()

list_cafe_filter=['Bar','Fast Food Restaurant']
indexNames = nearby_venues_clean[ (nearby_venues_clean['Venue_Main_Category'] == 'Café') & (nearby_venues_clean['Venue_Category'].isin(list_cafe_filter)) ].index
nearby_venues_clean.drop(indexNames , inplace=True)

list_Gym_filter=['Hotel','Residential Building (Apartment / Condo)','Martial Arts Dojo','Building','Hotel Pool','Track']
indexNames = nearby_venues_clean[ (nearby_venues_clean['Venue_Main_Category'] == 'Gym') & (nearby_venues_clean['Venue_Category'].isin(list_Gym_filter)) ].index
nearby_venues_clean.drop(indexNames , inplace=True)

list_Food_filter=['Café']
indexNames = nearby_venues_clean[ (nearby_venues_clean['Venue_Main_Category'] == 'Other Food') & (nearby_venues_clean['Venue_Category'].isin(list_Food_filter)) ].index
nearby_venues_clean.drop(indexNames , inplace=True)


for idx,cat_m in nearby_venues_clean['Venue_Main_Category'].iteritems():
    cat=nearby_venues_clean.loc[idx,'Venue_Category']
    if cat_m=='Café' or cat_m=='Gym':
        nearby_venues_clean.loc[idx,'Final_Category']=cat_m
    else:
        nearby_venues_clean.loc[idx,'Final_Category']=cat
    
nearby_venues_clean.head()
#nearby_venues_clean[nearby_venues_clean['Area']=='Raffles Place Area'].groupby(['Venue_Main_Category'])['Venue_Category'].value_counts(normalize=False)
venue_freq=pd.crosstab(nearby_venues_clean.Area,nearby_venues_clean.Venue_Main_Category,margins=False)
venue_freq.sort_values('Gym',ascending=False)
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr, _ = pearsonr(venue_freq["Café"], venue_freq["Gym"])
print('Pearsons correlation: %.3f' % corr)
shenton=nearby_venues_clean[nearby_venues_clean['Area']=='Shenton Way Area']
shenton.head()
geo = Nominatim(user_agent='My-IBMNotebook')
address = 'Singapore'
location = geo.geocode(address)
latitude = location.latitude
longitude = location.longitude

# create map
map_shenton = folium.Map(location=[latitude, longitude], tiles="Openstreetmap", zoom_start=11)

# set color scheme for the clusters
x = np.arange(3)
ys = [i+x+(i*x)**2 for i in range(3)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
cluster=0
for lat, lon, poi, cat in zip(shenton['Venue_Latitude'], shenton['Venue_Longitude'], shenton['Venue'], shenton['Venue_Main_Category']):
    if cat=='Gym':
        cluster=0
        color='green'
    elif cat=='Café':
        cluster=1
        color='red'
    else:
        cluster=2
        color='yellow'
        
    label = folium.Popup(str(cat) + '-' + str(poi), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=6,
        popup=label,
        color=color,
        #color=rainbow[cluster-1],
        fill=True,
        fill_color=color,
        #fill_color=rainbow[cluster-1],
        fill_opacity=1).add_to(map_shenton)
       
map_shenton
shenton_gym=shenton[shenton['Venue_Main_Category']=='Gym']
venue_list=[]
for idx,lat in shenton_gym['Venue_Latitude'].iteritems():
    lng=shenton_gym.loc[idx,'Venue_Longitude']
    name=shenton_gym.loc[idx,'Venue']
    sll=str(lat) + ',' + str(lng)
    params = dict(
      client_id=cid,
      client_secret=csecret,
      v='20180323',
      ll=sll,
      radius=50,
      query='Café'
    )
    resp = requests.get(url=url, params=params).json()["response"]['groups'][0]['items']
    main_category="Café"
    venue_list.append([(name,lat,lng,main_category,v['venue']['id'],v['venue']['name'],v['venue']['location']['lat'],v['venue']['location']['lng'],v['venue']['categories'][0]['name']) for v in resp])
for idx,lat in shenton_gym['Venue_Latitude'].iteritems():
    lng=shenton_gym.loc[idx,'Venue_Longitude']
    name=shenton_gym.loc[idx,'Venue']
    sll=str(lat) + ',' + str(lng)
    params = dict(
      client_id=cid,
      client_secret=csecret,
      v='20180323',
      ll=sll,
      radius=50,
      query='food'
    )
    resp = requests.get(url=url, params=params).json()["response"]['groups'][0]['items']
    main_category="Other food"
    venue_list.append([(name,lat,lng,main_category,v['venue']['id'],v['venue']['name'],v['venue']['location']['lat'],v['venue']['location']['lng'],v['venue']['categories'][0]['name']) for v in resp])
for idx,lat in shenton_gym['Venue_Latitude'].iteritems():
    lng=shenton_gym.loc[idx,'Venue_Longitude']
    name=shenton_gym.loc[idx,'Venue']
    sll=str(lat) + ',' + str(lng)
    params = dict(
      client_id=cid,
      client_secret=csecret,
      v='20180323',
      ll=sll,
      radius=50,
      query='Gym'
    )
    resp = requests.get(url=url, params=params).json()["response"]['groups'][0]['items']
    main_category="Gym"
    venue_list.append([(name,lat,lng,main_category,v['venue']['id'],v['venue']['name'],v['venue']['location']['lat'],v['venue']['location']['lng'],v['venue']['categories'][0]['name']) for v in resp])
nearby_shenton = pd.DataFrame([item for venue_list in venue_list for item in venue_list])
nearby_shenton.columns = ['Venue','Gym_Latitude','Gym_Longitude','Venue_Main_Category','Venue_ID','Sub_Venue','Venue_Latitude','Venue_Longitude','Venue_Category']

nearby_shenton.head()
nearby_shenton.groupby(['Venue_Main_Category'])['Venue_Category'].value_counts(normalize=False)
indexNames = nearby_shenton[ (nearby_shenton['Venue_Category'] == 'Residential Building (Apartment / Condo)')].index
nearby_shenton.drop(indexNames , inplace=True)
nearby_shenton.groupby(['Venue_Main_Category'])['Venue_Category'].value_counts(normalize=False)

list_Food_filter=['Café']
indexNames = nearby_shenton[ (nearby_shenton['Venue_Main_Category'] == 'Other Food') & (nearby_shenton['Venue_Category'].isin(list_Food_filter)) ].index
nearby_shenton.drop(indexNames , inplace=True)
nearby_shenton.head()
shenton_gym_freq=pd.crosstab(nearby_shenton.Venue,nearby_shenton.Venue_Main_Category,margins=False)
shenton_gym_freq.sort_values('Gym',ascending=False)
shenton_gym[['Venue','Venue_Latitude','Venue_Longitude']]
shenton_gym_freq2=pd.merge(shenton_gym_freq,shenton_gym[['Venue','Venue_Latitude','Venue_Longitude']],on='Venue',how='left')
shenton_gym_freq2
shenton_gym_freq3=shenton_gym_freq2.drop('Venue', 1)
kclusters = 4
# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=1).fit(shenton_gym_freq3)

# check cluster labels generated for each row in the dataframe
print(kmeans.labels_[0:30])
print(len(kmeans.labels_))
shenton_gym_freq2['Cluster_Labels'] = kmeans.labels_
shenton_gym_freq2.sort_values('Cluster_Labels',ascending=False)
shenton=shenton_gym_freq2
geo = Nominatim(user_agent='My-IBMNotebook')
address = 'Singapore'
location = geo.geocode(address)
latitude = location.latitude
longitude = location.longitude

# create map
map_shenton2 = folium.Map(location=[latitude, longitude], tiles="Openstreetmap", zoom_start=11)

# set color scheme for the clusters
x = np.arange(4)
ys = [i+x+(i*x)**2 for i in range(4)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(shenton['Venue_Latitude'], shenton['Venue_Longitude'], shenton['Venue'], shenton['Cluster_Labels']):        
    label = folium.Popup(str(poi) + ' cluster:' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=8,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=1).add_to(map_shenton2)
       
map_shenton2
