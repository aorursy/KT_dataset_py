import pandas as pd
pd.set_option('max_colwidth', 800)
import numpy as np

from bs4 import BeautifulSoup
import requests

from geopy.geocoders import Nominatim

import folium 
from pandas.io.json import json_normalize

import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans
url = 'https://en.wikipedia.org/wiki/List_of_districts_of_Jakarta'
res = requests.get(url).text 
soup = BeautifulSoup(res, 'html.parser')
lis = soup.find_all('li')
lis = lis[5:51] # Removing the redundant elements
neighborhoods = []
districts = []

for li in lis:
    li = li.get_text() # Getting the text from <li> elements
    neighborhoods.append(li)
for neighborhood in neighborhoods:
    if 'Jakarta' in neighborhood:
        districts.append(neighborhood)
        neighborhoods.remove(neighborhood)

neighborhoods[:5]
df = pd.DataFrame(neighborhoods)
df.columns = ['Neighborhood']
df = df.iloc[5:32].reset_index().drop('index',axis=1)

print(df.shape)
df.head()
geolocator = Nominatim(user_agent="jakarta_explore")
latitudes = []
longitudes = []

for neighborhood in df['Neighborhood']:
    location = geolocator.geocode(neighborhood)
    latitudes.append(location.latitude)
    longitudes.append(location.longitude)
df['Longitude'] = longitudes
df['Latitude'] = latitudes
df.head()
# Jakarta latitude and longitude
jkt_lat = -6.200000
jkt_long = 106.816666

# Create the map of Jakarta
map_jakarta = folium.Map(location=[jkt_lat, jkt_long], zoom_start=11)

# Add markers to map
for lat, lng, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
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
        parse_html=False).add_to(map_jakarta)  
    
map_jakarta
foursquare_api = pd.read_csv('../input/foursquarecredential/foursquare-api.csv')
CLIENT_ID = foursquare_api['CLIENT_ID'].values[0] # replace with your Foursquare ID
CLIENT_SECRET = foursquare_api[' CLIENT_SECRET'].values[0] # replace with your Foursquare Secret
VERSION = '20200605'
LIMIT = 100
radius = 500

venues = []

for neighborhood, lat, long in zip(df['Neighborhood'], df['Latitude'], df['Longitude']):
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        long,
        radius, 
        LIMIT)
    
    results = requests.get(url).json()['response']['groups'][0]['items']
    
    for venue in results:
        venues.append((
            neighborhood,
            lat, 
            long, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))
venues_df = pd.DataFrame(venues)
venues_df.columns = ['Neighborhood', 'Neighborhood Latitude', 'Neighborhood Longitude', 'Venue Name', 'Venue Latitude', 'Venue Longitude', 'Venue Category']
print(venues_df.shape)
venues_df.head()
# One hot encoding
jakarta_onehot = pd.get_dummies(venues_df[['Venue Category']], prefix="", prefix_sep="")

# Completing the dataframe
jakarta_onehot['Neighborhood'] = venues_df['Neighborhood'] 

# Moving neighborhood column to the first column
fixed_columns = list(jakarta_onehot.columns[96:]) + list(jakarta_onehot.columns[:96])
jakarta_onehot = jakarta_onehot[fixed_columns]

print(jakarta_onehot.shape)
jakarta_onehot.head()
jakarta_grouped = jakarta_onehot.groupby('Neighborhood').mean().reset_index()
jakarta_grouped.head()
# Function to return the most commong venues

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
num_top_venues = 10
indicators = ['st', 'nd', 'rd']

# Create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# Create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighborhood'] = jakarta_grouped['Neighborhood']

for ind in np.arange(jakarta_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(jakarta_grouped.iloc[ind, :], num_top_venues)
    
neighbourhoods_venues_sorted.head()
jakarta_clustering = jakarta_grouped.drop('Neighborhood', axis = 1)
max_range = 10 # Maximum range of clusters
from sklearn.metrics import silhouette_samples, silhouette_score

indices = []
scores = []

for kclusters in range(2, max_range) :
    
    # Run k-means clustering
    jc = jakarta_clustering
    kmeans = KMeans(n_clusters = kclusters, init = 'k-means++', random_state = 420).fit_predict(jc)
    
    # Gets the score for the clustering operation performed
    score = silhouette_score(jc, kmeans)
    
    # Appending the index and score to the respective lists
    indices.append(kclusters)
    scores.append(score)
import matplotlib.pyplot as plt
%matplotlib inline

def plot(x, y, xlabel, ylabel):
    plt.figure(figsize=(20,10))
    plt.plot(np.arange(2, x), y, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(2, x))
    plt.show()
plot(max_range, scores, "No. of clusters", "Silhouette Score")
opt = np.argmax(scores) + 2 # Finds the optimal value
opt
kclusters = opt

# Run k-means clustering
jc = jakarta_clustering
kmeans = KMeans(n_clusters = kclusters, init = 'k-means++', random_state = 0).fit(jc)
# Add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
neighbourhoods_venues_sorted.head()
jakarta_final = df
jakarta_final = jakarta_final.join(neighbourhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')
jakarta_final.dropna(inplace = True)
jakarta_final['Cluster Labels'] = jakarta_final['Cluster Labels'].astype(int)
jakarta_final.head()
# Create map
map_clusters = folium.Map(location=[jkt_lat, jkt_long], zoom_start=11)

# Set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(jakarta_final['Latitude'], jakarta_final['Longitude'], jakarta_final['Neighborhood'], jakarta_final['Cluster Labels']):
    label = folium.Popup(str(poi) + ' (Cluster ' + str(cluster + 1) + ')', parse_html=True)
    map_clusters.add_child(
        folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7))
       
map_clusters
clusters = pd.DataFrame(jakarta_final.groupby('Cluster Labels', as_index=False).apply(lambda x: ", ".join(x['Neighborhood'].tolist())), columns=['Similar'])
clusters['Count'] = jakarta_final.groupby('Cluster Labels')['Neighborhood'].count()

cluster_0 = jakarta_final.loc[jakarta_final['Cluster Labels']==0].Neighborhood.values
cluster_1 = jakarta_final.loc[jakarta_final['Cluster Labels']==1].Neighborhood.values
cluster_2 = jakarta_final.loc[jakarta_final['Cluster Labels']==2].Neighborhood.values
cluster_3 = jakarta_final.loc[jakarta_final['Cluster Labels']==3].Neighborhood.values

print("Cluster | Count | Neighbourhoods")
print("--------|-------|---------------")
cnt = 0
for x,y,z in zip(clusters.index,clusters['Count'],clusters['Similar']):
    if cnt == 0:
        z = ', '.join(cluster_0)
    elif cnt == 1:
        z = ', '.join(cluster_1)
    elif cnt == 2:
        z = ', '.join(cluster_2)
    else:
        z = ', '.join(cluster_3)
    cnt += 1
    print("{} | {} | {}".format(x,y,z))
jakarta_final.head()
clusters_grouped = jakarta_final.groupby(['Cluster Labels','1st Most Common Venue']).count().reset_index()
clusters_grouped[['Cluster Labels', '1st Most Common Venue']]