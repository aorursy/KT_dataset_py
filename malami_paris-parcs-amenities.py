import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

from sklearn.cluster import KMeans

import matplotlib.cm as cm
import matplotlib.colors as colors

print('Folium installed')
print('Libraries imported.')
# path of data 
path = "../input/parcs.csv"
df = pd.read_csv(path, sep=';')

latitude_longitude = df["geo_point_2d"].str.split(",", n = 1, expand = True)
latitude = latitude_longitude[0]
longitude = latitude_longitude[1]

df.insert(3, "latitude", latitude )
df.insert(4, "longitude", longitude ) 
df[["latitude", "longitude"]] = df[["latitude", "longitude"]].apply(pd.to_numeric)
#df=df.head(30)

df[["ID", "NOM_PARC", "latitude", 'longitude']].head()

# You can get position of Paris from this code segement:

#address = 'Paris, France'
#geolocator = Nominatim(user_agent="foursquare_agent")
#location = geolocator.geocode(address)
#latitude = location.latitude
#longitude = location.longitude
#print(latitude, longitude)

latitude = 48.847687
longitude = 2.303116
# create map of Paris using latitude and longitude values
map_paris = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, ID, NOM_PARC in zip(df['latitude'], df['longitude'], df['ID'], df['NOM_PARC']):
    label = '{}, {}'.format(NOM_PARC, ID)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_paris)  
    
map_paris
CLIENT_ID = 'XGBSKVV0X2RAQBFN53N52QPZX0NA0F0QRUHL5JIO3LSTBE43' # your Foursquare ID
CLIENT_SECRET = 'R03IY1GXPGTPU2DISXDELGZVXWV1M13OE0DE4DGCMJTCX44L' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 100

def getNearbyVenues(names, latitudes, longitudes, radius=500):

    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name, lat, lng)
            
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
        
        #print(results)
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
parc_venues = getNearbyVenues(names = df['ID'], latitudes = df['latitude'], longitudes = df['longitude'])
parc_venues.shape
parc_venues.head()
print('There are {} uniques categories.'.format(len(parc_venues['Venue Category'].unique())))
df_unified = parc_venues.copy()
matching_table = placard = {
    "shop":"shop",
    "Market":"shop",
    "Supermarket":"shop",
    "Hotel": "Hotel", 
    "motel": "Hotel",
    "Resort": "Hotel",
    "Restaurant": "Restaurant",
    "creperie": "Restaurant",
     "Pizza":"Restaurant",
    "Diner":"Restaurant",
    "Food":"Restaurant",
    "Breakfast":"Restaurant",
    "bar":"bar",
    "Brasserie":"bar",
    "club":"Entertainment",
    "SPORT": "Entertainment",
    "Plaza":"Entertainment",
    "Theater":"Entertainment",
    "Stadium":"Entertainment",
    "Tennis":"Entertainment",
    "Music":"Entertainment",
    "Fitness":"Entertainment",
    "Cultural":"Entertainment",
    "Pool":"Entertainment",
    "Studio":"Entertainment",
    "Concert":"Entertainment",
    "Basketball":"Entertainment",
    "Beauty":"Entertainment",
    "House":"Entertainment",
    "Arcade":"Entertainment",
    "Auditorium":"Entertainment",
    "Aquarium":"Entertainment",
    "Photography":"Entertainment",
    "Gym":"Entertainment",
    "Art":"Galerie",
    "café":"café",
    "cafe":"café",
    "Tea":"café",
    "store":"store",
    "tram":"Transtport",
    "bus":"Transtport",
    "train":"Transtport",
    "bike":"Transtport",
    "Office":"Administration"
}


for key, value in matching_table.items():
    df_unified.loc[df_unified['Venue Category'].str.upper().str.contains(key.upper()), 'Venue Category'] = matching_table[key].upper()
    matching_table[key] = matching_table[key].upper()
    

df_filtred = df_unified.loc[df_unified['Venue Category'].isin(matching_table.values())].copy()

df_filtred.groupby("Venue Category").count()
print('There are {} uniques categories.'.format(len(df_filtred['Venue Category'].unique())))
df_filtred['Venue Category'].unique()
# one hot encoding
parc_onehot = pd.get_dummies(df_filtred[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
parc_onehot['Neighborhood'] = df_filtred['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [parc_onehot.columns[-1]] + list(parc_onehot.columns[:-1])
parc_onehot = parc_onehot[fixed_columns]

parc_onehot.head()
save_parcgrouped = parc_onehot
parc_grouped = parc_onehot.groupby('Neighborhood').mean().reset_index()
parc_grouped
num_top_venues = 10

for hood in parc_grouped['Neighborhood']:
    temp = parc_grouped[parc_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
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
neighborhoods_venues_sorted['Neighborhood'] = parc_grouped['Neighborhood']

for ind in np.arange(parc_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(parc_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
parc_grouped_clustering = parc_grouped.drop('Neighborhood', 1)
parc_grouped_clustering.head()
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(parc_grouped_clustering)
    Sum_of_squared_distances.append(km.inertia_)
    

Sum_of_squared_distances
import matplotlib.pyplot as plt
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
# set number of clusters
kclusters = 5

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(parc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

parc_merged = pd.DataFrame(df[["ID", "NOM_PARC", "latitude", "longitude"]])

# merge parc_merged with neighborhoods_venues_sorted to add latitude/longitude for each neighborhood
parc_merged = parc_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='ID')

parc_merged.head() # check the last columns!

save_parcgrouped = save_parcgrouped[['Neighborhood', 'ADMINISTRATION', 'BAR', 'CAFÉ', 'ENTERTAINMENT', 'GALERIE', 'HOTEL', 'RESTAURANT', 'SHOP', 'STORE', 'TRANSTPORT']]
save_parcgrouped = save_parcgrouped.join(parc_merged.set_index('ID'), on='Neighborhood')
save_parcgrouped = save_parcgrouped[['Cluster Labels', 'ADMINISTRATION', 'BAR', 'CAFÉ', 'ENTERTAINMENT',
       'GALERIE', 'HOTEL', 'RESTAURANT', 'SHOP', 'STORE', 'TRANSTPORT']]
save_parcgrouped = save_parcgrouped.groupby('Cluster Labels').sum().reset_index(drop=True)
save_parcgrouped.head()
save_parcgrouped = save_parcgrouped.T
save_parcgrouped.columns = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
print(save_parcgrouped)
save_parcgrouped[['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']] = round(save_parcgrouped[['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']].apply(lambda x: x/x.sum(), axis=1), 2)
save_parcgrouped[['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']]
ax=save_parcgrouped[['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']].plot(
    kind="bar", figsize=(25, 8), rot=90, fontsize=14,  
    colors = ['#8cc85c', '#8b70de', '#89534f', '#9cb85c', '#5bc0de', '#d9534f'], width=0.6)
plt.title("Percentage of the categories repartition for each cluster", fontsize=16)
plt.yticks([])
ax.set_xticklabels(save_parcgrouped.index)
 
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height), (x + width /7, y + height + 0.01))
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.show()
address = 'Paris, France'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(parc_merged['latitude'], parc_merged['longitude'], parc_merged['ID'], parc_merged['Cluster Labels']):
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
