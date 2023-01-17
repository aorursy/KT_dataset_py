import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import json # library to handle JSON files




import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML and XML documents

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import folium # map rendering library

print("Libraries imported.")
data=requests.get('https://en.wikipedia.org/wiki/List_of_neighbourhoods_in_Mumbai').text
soup=BeautifulSoup(data,'html.parser')
area=[]
loc=[]
lat=[]
lon=[]
for row in soup.find('table').find_all('tr'):
    cells=row.find_all('td')
    if (len(cells)>0):
        area.append(cells[0].text)
        loc.append(cells[1].text)
        lat.append(cells[2].text)
        lon.append(cells[3].text)
area_mumbai=[]
for areas in area:
    area_mumbai.append(areas.replace('\n',''))
area_mumbai[0:5]
loc_mumbai=[]
for locations in loc:
    loc_mumbai.append(locations.replace('\n',''))
loc_mumbai[0:5]
lat_mumbai=[]
for lats in lat:
    lat_mumbai.append(lats.replace('\n',''))
lon_mumbai=[]
for lons in lon:
    lon_mumbai.append(lons.replace('\n',''))
lat_mumbai[0:5]
lon_mumbai[0:5]
df_mumbai=pd.DataFrame(columns=['Area','Location','Latitude','Longitude'])
df_mumbai['Area']=area_mumbai
df_mumbai['Location']=loc_mumbai
df_mumbai['Latitude']=lat_mumbai
df_mumbai['Longitude']=lon_mumbai
df_mumbai
df_mumbai['Longitude'][82]=72.8479
df_mumbai
df_mumbai.to_csv('Mumbai neighborhood coordinates.csv')
latitude=19.07 
longitude=72.87

map_mumbai = folium.Map(location=[latitude, longitude], zoom_start=11)

for lat,lon,areas,location in zip(df_mumbai['Latitude'],df_mumbai['Longitude'],df_mumbai['Area'],df_mumbai['Location']):
                                        
                                        label='{} {}'.format(areas,location)
                                        label=folium.Popup(label)
                                        
                                        folium.CircleMarker(
                                            [lat,lon], radius=5,popup=label,color='orange',fill=True,fill_color='black',fill_opacity=0.6).add_to(map_mumbai)
                                   
map_mumbai
CLIENT_ID='Your client ID'
CLIENT_SECRET='Your client secret'
VERSION = '20180605'
venues = []

radius = 1000
LIMIT = 100


for lat, lon, loc,areas in zip(df_mumbai['Latitude'], df_mumbai['Longitude'], df_mumbai['Location'], df_mumbai['Area']):
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        lon,
        radius, 
        LIMIT)
    
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    for venue in results:
        venues.append((
            areas, 
            loc,
            lat, 
            lon, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))
venues_df=pd.DataFrame(venues)
venues_df.rename(columns={0:'Area',1:'Location',2:'Area latitude',3:'Area longitude',4:'Venue name',5:'Venue latitude',6:'Venue longitude',7:'Venue category'},inplace=True)
venues_df.head()
venues_df['Venue category'].unique()
len(venues_df['Venue category'].unique())
categories_onehot=pd.get_dummies(venues_df['Venue category'])
mumbai_category_df=pd.DataFrame(venues_df['Area'],columns=['Area'])
mumbai_category_df=mumbai_category_df.merge(categories_onehot,on=mumbai_category_df.index)
mumbai_category_df.head()
mumbai_category_grouped=mumbai_category_df.groupby(['Area']).mean().reset_index()

mumbai_category_grouped.drop('key_0',axis=1,inplace=True)
mumbai_category_grouped.head()
bar_list=['Sports Bar','Gastropub','Bar','Beer Bar',
          'Beer Garden','Club House',
          'Lounge','Cocktail Bar','Hotel Bar',
          'Bistro','Brewery','Wine Bar','Nightclub']
bar_category_df=pd.DataFrame(columns=[mumbai_category_grouped.columns])
bar_category_df=mumbai_category_grouped[mumbai_category_grouped['Pub']>0]

for i in range(0,len(bar_list)):
    bar_category_df=bar_category_df.append(mumbai_category_grouped[mumbai_category_grouped['{}'.format(bar_list[i])]>0])
bar_category_df.reset_index(drop=True,inplace=True)
bar_category_df.head()
wcss=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i,max_iter=300)
    kmeans.fit(bar_category_df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.ylabel('WCSS')
plt.xlabel('K clusters')
plt.xticks(np.arange(1,15))
plt.axvline(5,color='red')

k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(bar_category_df.iloc[:,1:])
labels=kmeans.labels_
cluster_df=pd.DataFrame(columns=['Area','Label'])
cluster_df['Area']=bar_category_df.iloc[:,0]

cluster_df['Label']=labels
cluster_df.head()
cluster_df=cluster_df.merge(df_mumbai,on='Area')
cluster_df
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(k)
ys = [i+x+(i*x)**4 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0,1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, area,loc,cluster in zip(cluster_df['Latitude'], cluster_df['Longitude'], cluster_df['Area'], cluster_df['Location'],cluster_df['Label']):
    label = folium.Popup('{} ({}) - Cluster {}'.format(area,loc,cluster+1), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=6,
        popup=label,
        color=rainbow[cluster-2],
        fill=True,
        fill_color=rainbow[cluster-2],
        fill_opacity=0.8).add_to(map_clusters)
       
map_clusters
cluster_df[cluster_df['Label']==0]
cluster_df[cluster_df['Label']==1]
cluster_df[cluster_df['Label']==2]
cluster_df[cluster_df['Label']==3]
cluster_df[cluster_df['Label']==4]
sizes=[]

for labels in np.arange(0,5):
    sizes.append(cluster_df[cluster_df['Label']==labels].shape[0])
sizes_df=pd.DataFrame(columns=['Label name','Label size'])
sizes_df['Label name']=np.arange(1,6)
sizes_df['Label size']=sizes
sizes_df.index=sizes_df['Label name']
sizes_df.drop('Label name', axis=1,inplace=True)
sizes_df
