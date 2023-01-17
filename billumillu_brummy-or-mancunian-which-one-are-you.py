import pandas as pd

import numpy as np

import requests

from pandas.io.json import json_normalize

from sklearn.cluster import KMeans

from matplotlib import cm

from matplotlib import colors



from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter



# !conda install -c conda-forge folium=0.5.0 --yes

import folium
# @hidden_cell

CLIENT_ID = 'JTFC2YRZFN5HFUVSE5PXBOEPFOLWQO02VFGOC42KUA13SRLM' # your Foursquare ID

CLIENT_SECRET = 'KL1JALAN1PY4XO0CGAD2IMZB153UM1THKPUOG0B22OU04VRV' # your Foursquare Secret

VERSION = '20180604'

LIMIT = 30



print("Foursquare credentials initialized")
list=[]

dfs = pd.read_html('https://en.wikipedia.org/wiki/B_postcode_area',header=0)

bir = dfs[1]

bir.head()
bir['Post town'].value_counts()
bir = bir[bir['Post town'] == 'BIRMINGHAM']

bir['Post town'].value_counts()
bir.isnull().sum()
bir.dropna(subset=['Coverage'],inplace=True)

bir.isnull().sum()
locator = Nominatim(user_agent="myGeocoder")



# 1 - conveneint function to delay between geocoding calls

geocode = RateLimiter(locator.geocode, min_delay_seconds=1)



# 2- - create location column

bir['location'] = bir['Coverage'].apply(geocode)
# 3 - create longitude, laatitude and altitude from location column (returns tuple)

bir['point'] = bir['location'].apply(lambda loc: tuple(loc.point) if loc else None)



# 4 - drop null values

bir.dropna(subset=['location','point'],inplace=True)



# 5 - split point column into latitude, longitude and altitude columns

bir[['latitude', 'longitude', 'altitude']] = pd.DataFrame(bir['point'].tolist(), index=bir.index)
bir.head(3)
bir.isnull().sum()
bir.drop(columns=['Postcode district','Post town','Local authority area','location','point','altitude'], inplace=True)

bir.rename({'Coverage':'Neighborhood'},axis=1,inplace=True)
bir.head()
list=[]

dfs = pd.read_html('https://en.wikipedia.org/wiki/M_postcode_area',header=0)

man = dfs[1]

man.head()
man['Post town'].value_counts()
man = man[man['Post town'] == 'MANCHESTER']

man['Post town'].value_counts()
man.isnull().sum()
man['location'] = man['Coverage'].apply(geocode)

man['point'] = man['location'].apply(lambda loc: tuple(loc.point) if loc else None)

man[['latitude', 'longitude', 'altitude']] = pd.DataFrame(man['point'].tolist(), index=man.index)
man.drop(columns=['Postcode district','Post town','Local authority area','location','point','altitude'], inplace=True)

man.head(3)
man.isnull().sum()
man.dropna(subset=['latitude','longitude'],inplace=True)

man.isnull().sum()
man.rename({'Coverage':'Neighborhood'},axis=1,inplace=True)
man.head()
def getNearbyVenues(names, latitudes, longitudes, radius=500):

    

    venues_list=[]

    for name, lat, lng in zip(names, latitudes, longitudes):

            

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

        results = requests.get(url).json()['response']['groups'][0]['items']

        

        # return only relevant information for each nearby venue

        venues_list.append([( 

            v['venue']['name'],

            v['venue']['categories'][0]['name']) for v in results])



    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

    nearby_venues.columns = ['Venue','Venue Category']

    

    return(nearby_venues)
bir_venues = getNearbyVenues(names=bir['Neighborhood'],

                             latitudes=bir['latitude'],

                             longitudes=bir['longitude']

                            )
bir_venues.head()
print("Number of places of interest:", format(bir_venues.shape[0]))
bir_top = bir_venues['Venue Category'].value_counts()[0:10]

bir_top
man_venues = getNearbyVenues(names=man['Neighborhood'],

                             latitudes=man['latitude'],

                             longitudes=man['longitude']

                            )
man_venues.head()
print("Number of places of interest:", format(man_venues.shape[0]))
man_top = man_venues['Venue Category'].value_counts()[0:10]

man_top
a = (bir_venues['Venue Category']=="Zoo").sum(), (man_venues['Venue Category']=="Zoo").sum() 

print("Number of zoos in Birmingham and Manchester: ",a)
bir_onehot = pd.get_dummies(bir_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

bir_onehot['Neighborhood'] = bir['Neighborhood'] 



bir_onehot.set_index("Neighborhood",inplace=True)



bir_onehot.head(3)
# Group by neighborhood

bir_grouped = bir_onehot.groupby('Neighborhood').mean().reset_index()



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

neighborhoods_venues_sorted['Neighborhood'] = bir_grouped['Neighborhood']



for ind in np.arange(bir_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(bir_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted.head(3)
kclusters = 3



bir_grouped_clustering = bir_grouped.drop('Neighborhood', 1)



kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(bir_grouped_clustering)



kmeans.labels_
# New dataframe that includes the cluster as well as the top 10 venues for each neighborhood



# Drop old 'Cluster Labels' column if it exists

if "Cluster Labels" in neighborhoods_venues_sorted.columns:

    neighborhoods_venues_sorted = neighborhoods_venues_sorted.drop('Cluster Labels', axis=1)



# add clustering labels

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)



bir_merged = bir



# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

bir_merged = bir_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



bir_merged.head(3)
bir_merged.set_index('Neighborhood',inplace=True)

bir_merged.drop(['Handsworth','Yardley'],inplace=True)

bir_merged.reset_index(inplace=True)

bir_merged.head(2)
# Birmingham's coordinates

lat = 52.4862

lon = -1.8904

map_clusters = folium.Map(location=[lat,lon], zoom_start=10)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(bir_merged['latitude'], bir_merged['longitude'], bir_merged['Neighborhood'], bir_merged['Cluster Labels']):

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
b1 = bir_merged.loc[bir_merged['Cluster Labels'] == 0, :]

b1 = b1[['Neighborhood','1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue']]

b1
b2 = bir_merged.loc[bir_merged['Cluster Labels'] == 1, :]

b2 = b2[['Neighborhood','1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue']]

b2
b3 = bir_merged.loc[bir_merged['Cluster Labels'] == 2, :]

b3 = b3[['Neighborhood','1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue']]

b3
man_onehot = pd.get_dummies(man_venues[['Venue Category']], prefix="", prefix_sep="")

man_onehot['Neighborhood'] = man['Neighborhood'] 

man_onehot.set_index("Neighborhood",inplace=True)

man_grouped = man_onehot.groupby('Neighborhood').mean().reset_index()

def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)   

    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

columns = ['Neighborhood']

for ind in np.arange(num_top_venues):

    try:

        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common Venue'.format(ind+1))

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)

neighborhoods_venues_sorted['Neighborhood'] = man_grouped['Neighborhood']

for ind in np.arange(man_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(man_grouped.iloc[ind, :], num_top_venues)



kclusters = 3

man_grouped_clustering = man_grouped.drop('Neighborhood', 1)

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(man_grouped_clustering)



if "Cluster Labels" in neighborhoods_venues_sorted.columns:

    neighborhoods_venues_sorted = neighborhoods_venues_sorted.drop('Cluster Labels', axis=1)

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

man_merged = man

man_merged = man_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



# Manchester's coordinates

lat = 53.4808

lon = -2.2426

map_clusters = folium.Map(location=[lat,lon], zoom_start=10)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(man_merged['latitude'], man_merged['longitude'], man_merged['Neighborhood'], man_merged['Cluster Labels']):

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
m1 = man_merged.loc[man_merged['Cluster Labels'] == 0, :]

m1 = m1[['Neighborhood','1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue']]

m1
m2 = man_merged.loc[man_merged['Cluster Labels'] == 1, :]

m2 = m2[['Neighborhood','1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue']]

m2
m3 = man_merged.loc[man_merged['Cluster Labels'] == 2, :]

m3 = m3[['Neighborhood','1st Most Common Venue','2nd Most Common Venue','3rd Most Common Venue','4th Most Common Venue']]

m3