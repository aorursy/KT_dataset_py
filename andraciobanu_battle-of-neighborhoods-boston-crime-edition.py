# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json
with open('../input/boston-neighborhoods-geojson/Boston_Neighborhoods.geojson', 'r') as f:
    boston_geojson = json.load(f)
features = boston_geojson['features']
nbh_list = []
for feature in features:
    nbh_list.append(feature['properties']['Name'])
print(nbh_list)
from shapely.geometry import Point, shape, Polygon
column_names = ['Neighborhood', 'Latitude', 'Longitude'] 
boston_neighborhoods = pd.DataFrame(columns=column_names)
for feature in features:
        polygon = shape(feature['geometry'])
        neighborhood_name = feature['properties']['Name']
        boston_neighborhoods = boston_neighborhoods.append({'Neighborhood': neighborhood_name,
                                          'Latitude': polygon.centroid.y,
                                          'Longitude': polygon.centroid.x}, ignore_index=True)
boston_neighborhoods.head()
boston_neighborhoods.shape

import folium

map_boston = folium.Map(location=[42.361145, -71.057083], zoom_start=12)

# add markers to map
for lat, lng, neighborhood in zip(boston_neighborhoods['Latitude'], boston_neighborhoods['Longitude'], boston_neighborhoods['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_boston)  
    
map_boston

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

data_crime = pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin1')
data_crime.head()
#data_crime.set_index('INCIDENT_NUMBER', inplace = True)
data_crime['SHOOTING'].fillna(0, inplace = True)
data_crime.drop(columns = ['Location'], inplace = True)
data_crime.head()
data_crime.isna().sum()
data_crime_od = data_crime.groupby('OFFENSE_DESCRIPTION').size().reset_index(name = 'counts').set_index('OFFENSE_DESCRIPTION').sort_values(by = 'counts', ascending = False)
data_crime_od
df_ocg = data_crime.groupby('OFFENSE_CODE_GROUP').size().reset_index(name = 'counts').set_index('OFFENSE_CODE_GROUP').sort_values(by = 'counts', ascending = False)
df_ocg.head()
def multiliner(string_list, n):
    length = len(string_list)
    for i in range(length):
        rem = i % n
        string_list[i] = '\n' * rem + string_list[i]
    return string_list

fig41 = plt.figure(figsize = (20,8))
ind41 = np.arange(10)
ax41 = plt.subplot(111)
y_data = df_ocg['counts'].head(10)
df_riocg = df_ocg.reset_index()
rects = ax41.bar(ind41, y_data, width = 0.8,color = 'r')
ax41.set_xticks(ind41)
ax41.set_xticklabels(multiliner(df_ocg.index.tolist()[:10], 2))
ax41.set_xlabel('Offense Code Group')
ax41.set_ylabel('Amount of crimes')
ax41.set_title('Crimes in Boston by offense code group')
for rect in rects:
    height = rect.get_height()
    ax41.text(rect.get_x() + 0.2, 1.02 * height, height, fontsize = 14)
def point_to_neighborhood (lat, long, geojson):
    point = Point(long, lat)
    features = geojson['features']
    for feature in features:
        polygon = shape(feature['geometry'])
        neighborhood = feature['properties']['Name']
        if polygon.contains(point):
            if neighborhood == 'Chinatown' or neighborhood == 'Leather District':
                return 'Downtown'
            elif neighborhood == 'Bay Village':
                return 'South End'
            else:
                return neighborhood
    print(f'Point ({long},{lat}) is not in Boston.')
    return None
df_nafree = data_crime.dropna(subset = ['Lat','Long'])
df_nafree.shape
for index, row in df_nafree.iterrows():
    lat = df_nafree.at[index, 'Lat']
    long = df_nafree.at[index, 'Long']
    #print(index)
    #print(lat)
    #print(long)
    neighborhood = point_to_neighborhood(lat, long, boston_geojson)
    #print(neighborhood)
    df_nafree.at[index, 'Neighborhood'] = neighborhood
df_nafree.tail(10)
df_nbh = df_nafree.groupby('Neighborhood').size().reset_index(name = 'count').set_index('Neighborhood')


final=pd.merge(df_nbh,boston_neighborhoods,on='Neighborhood',sort=True)
final
latitudes=[]
longitudes=[]
for ind in final.index: 
    latitudes.append(final['Latitude'][ind])
    longitudes.append(final['Longitude'][ind])
    
    
import requests
import json
gov_category = '52e81612bcbc57f1066b7a38' 


gov_venues_categories = ['4bf58dd8d48988d12a941735','4bf58dd8d48988d129941735','4bf58dd8d48988d12b941735',
                        '4bf58dd8d48988d12c951735','4bf58dd8d48988d12c941735','4bf58dd8d48988d12d941735',
                        '4bf58dd8d48988d12e941735','52e81612bcbc57f1066b7a38']

foursquare_client_id='HTEBQOT1TLOIZW5PINS2NJ02UESJS4UM4WPDMELTXSUMU0NZ'
foursquare_client_secret='O4RHCRROMFSNIFEMWR4U5UGLRTP5SYUQOJKP0ESFZ20HGM2K'

def get_categories(categories):
    return [(cat['name'], cat['id']) for cat in categories]


def get_venues_near_location(lat, lon, category, client_id, client_secret, radius=500, limit=100):
    version = '20180724'
    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&categoryId={}&radius={}&limit={}'.format(
        client_id, client_secret, version, lat, lon, category, radius, limit)
    try:
        results = requests.get(url).json()['response']['groups'][0]['items']
        venues = [(item['venue']['id'],
                   item['venue']['name'],
                   get_categories(item['venue']['categories']),
                   (item['venue']['location']['lat'], item['venue']['location']['lng']),
                   format_address(item['venue']['location']),
                   item['venue']['location']['distance']) for item in results]        
    except:
        venues = []
    return venues



import requests
import json

def get_buildings(lats, lons):
    buildings = {}
    police_buildings = {}

    print('Obtaining venues around candidate locations:', end='')
    for lat, lon in lats, lons:
        venues = get_venues_near_location(lat, lon, gov_category, foursquare_client_id, foursquare_client_secret, radius=500, limit=100)
        for venue in venues:
            venue_id = venue[0]
            venue_name = venue[1]
            venue_categories = venue[2]
            venue_latlon = venue[3]
            venue_address = venue[4]
            venue_distance = venue[5]
            is_police=False
            if(venue[2]=='4bf58dd8d48988d12e941735'):
                is_police=True
            building = (venue_id, venue_name, venue_latlon[0], venue_latlon[1], venue_address, venue_distance, is_police)
            buildings[venue_id] = building
            if is_police:
                police_buildings[venue_id] = building
        print(' .', end='')
    print(' done.')
    return buildings, police_buildings

buildings, police_buildings= get_buildings(latitudes, longitudes)

final.to_csv('out.csv', index=False)

import pandas as pd
for_cluster=pd.read_csv('../input/hthkejgv/out.csv')
for_cluster
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


for_cluster_g = for_cluster.drop('Neighborhood', 1)

plt.plot()
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(for_cluster_g)
    kmeanModel.fit(for_cluster_g)
    distortions.append(sum(np.min(cdist(for_cluster_g, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / for_cluster_g.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

 

kclusters = 4

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(for_cluster_g)
for_cluster.insert(0, 'Cluster Labels', kmeans.labels_)

for_cluster
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors


map_clusters = folium.Map(location=[42.361145, -71.057083], zoom_start=10)

x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(for_cluster['Latitude'], for_cluster['Longitude'], for_cluster['Neighborhood'], for_cluster['Cluster Labels']):
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
print(for_cluster[for_cluster['Cluster Labels']==0]['count'].mean())
print(for_cluster[for_cluster['Cluster Labels']==0]['Gov_Buildings'].mean())
print(for_cluster[for_cluster['Cluster Labels']==1]['count'].mean())
print(for_cluster[for_cluster['Cluster Labels']==1]['Gov_Buildings'].mean())
print(for_cluster[for_cluster['Cluster Labels']==2]['count'].mean())
print(for_cluster[for_cluster['Cluster Labels']==2]['Gov_Buildings'].mean())
print(for_cluster[for_cluster['Cluster Labels']==3]['count'].mean())
print(for_cluster[for_cluster['Cluster Labels']==3]['Gov_Buildings'].mean())
recommended=pd.DataFrame(for_cluster[for_cluster['Cluster Labels']==2]['Neighborhood'])
recommended
