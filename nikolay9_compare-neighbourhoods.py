# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

print('Libraries imported.')
address = 'Toronto, Ontario'

#geolocator = Nominatim()
#location = geolocator.geocode(address)
latitude = 43.653963
longitude = -79.387207
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
table = pd.read_csv('../input/ibm-capstone-project-geodata/NeighborhodsToronto.csv',index_col = 0)
table.head()
# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, borough, neighborhood in zip(table['Latitude'], table['Longitude'], table['Borough'], table['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_toronto)  
    
map_toronto
CLIENT_ID = 'Foursquare ID' 
CLIENT_SECRET = 'Foursquare Secret' 
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
#print('CLIENT_ID: ' + CLIENT_ID)
#print('CLIENT_SECRET:' + CLIENT_SECRET)
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    LIMIT = 100
    venues_list=[]
    count = 0
    emtyIndices = []
    for name, lat, lng in zip(names, latitudes, longitudes):
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
        #this try except block for some neighborhoods which return None  venues 
        try:
            venues_list[count][0]
        except:
            emtyIndices.append(count)
            count-1
            
        count+=1     

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return nearby_venues, emtyIndices
#toronto_venues, empty =getNearbyVenues(table['Neighborhood'],table['Latitude'],table['Longitude'])
toronto_venues = pd.read_csv('../input/ibm-capstone-project-geodata/toronto_venues.csv')
empty = [16, 21, 93]
toronto_venues.groupby('Neighborhood').count()[:5]
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
toronto_onehot.shape
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped.head()
num_top_venues = 5

for hood in toronto_grouped['Neighborhood'][:5]:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
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
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head(10)
Pharmacy_top_10 = toronto_grouped.sort_values(by = 'Pharmacy', axis=0, ascending=False).drop(['Pharmacy'],axis = 1)[:10]
import math
def cosine_similarity_1(v1, v2):
    prod = np.dot(v1, v2)
    len1 = math.sqrt(np.dot(v1, v1))
    len2 = math.sqrt(np.dot(v2, v2))
    return prod / (len1 * len2)
for i in range(0,10):
    print("cosineSimilarity between\n {} and  {} \n== {}".format(Pharmacy_top_10.iloc[0,0],
                      Pharmacy_top_10.iloc[i,0],
                      cosine_similarity_1(Pharmacy_top_10.iloc[0,1:].
                         values,Pharmacy_top_10.iloc[i,1:].values)))
    print('*'*30)
        
def compare_Distance(neighborhood,top = Pharmacy_top_10,category = 'Pharmacy'):
    result = []
    for i in range(10):
        result.append(cosine_similarity_1(neighborhood.drop(labels=category).
                         values[1:],top.iloc[i,1:].values))
    return np.mean(result)
top_pharm_places = toronto_grouped.T.apply(compare_Distance)
top_pharm_places.sort_values(ascending = False)[:5]
top5neighbors = toronto_grouped.iloc[top_pharm_places.sort_values(ascending = False)[:5].index,:]['Neighborhood'].values
table[table['Neighborhood'].isin(top5neighbors)]
# set color scheme for the clusters
x = np.arange(6)
ys = [i+x+(i*x)**2 for i in range(6)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to the map
markers_colors = []
for lat, lon, poi in zip(table[table['Neighborhood'].isin(top5neighbors)]['Latitude'],
                         table[table['Neighborhood'].isin(top5neighbors)]['Longitude'],
                         table[table['Neighborhood'].isin(top5neighbors)]['Neighborhood']):
    label = folium.Popup(str(poi) + ' Pharmacy' , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[5],
        fill=True,
        fill_color=rainbow[5],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
toronto_grouped.iloc[top_pharm_places.sort_values(ascending = False)[:5].index,:]['Pharmacy']
toronto_grouped.sort_values(by = 'Pharmacy', axis=0, ascending=False)['Pharmacy'][:5]
def compare_distance_penalty(neighborhood,top = Pharmacy_top_10,category = 'Pharmacy'):
    result = []
    coefs = []
    for i in range(10):
        result.append(cosine_similarity_1(neighborhood.drop(labels=category).
                         values[1:],top.iloc[i,1:].values))
        coefs.append(neighborhood[category])
    coefs = np.array([coefs])*sorted(result, reverse=True)[0]
    return np.mean(np.array(result)-coefs)
top5_penalty = toronto_grouped.T.apply(compare_distance_penalty)
top5_penalty.sort_values(ascending = False)[:5]
toronto_grouped.iloc[top5_penalty.sort_values(ascending = False)[:5].index,:]['Pharmacy']
toronto_grouped.sort_values(by = 'Pharmacy', axis=0, ascending=False)['Pharmacy'][:5]
top5neighbors_names = toronto_grouped.iloc[top5_penalty.sort_values(ascending = False)[:5].index,:]['Neighborhood'].values
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to the map
markers_colors = []
for lat, lon, poi in zip(table[table['Neighborhood'].isin(top5neighbors_names)]['Latitude'],
                         table[table['Neighborhood'].isin(top5neighbors_names)]['Longitude'],
                         table[table['Neighborhood'].isin(top5neighbors_names)]['Neighborhood']):
    label = folium.Popup(str(poi) + ' Pharmacy with penalty' , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[0],
        fill=True,
        fill_color=rainbow[0],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
def find_best_places(data,grouped_data,category,top_count = 10,top_places = 5):
    category_top = grouped_data.sort_values(by = category, axis=0, ascending=False).drop([category],axis = 1)[:top_count]
    def compare_Distance(neighborhood,top = category_top,category = category):
        result = []
        for i in range(10):
            result.append(cosine_similarity_1(neighborhood.drop(labels=category).
                             values[1:],top.iloc[i,1:].values))
        return np.mean(result)
    top_places_found = grouped_data.T.apply(compare_Distance)
    top5neighbors = grouped_data.iloc[top_places_found.sort_values(ascending = False)[:5].index,:]['Neighborhood'].values
    frequncy = grouped_data.iloc[top_places_found.sort_values(ascending = False)[:5].index,:][category]#
    # set color scheme for the clusters
    
    places = data[data['Neighborhood'].isin(top5neighbors)]
    return top5neighbors, places, frequncy
top5neighbors, places, frequncy = find_best_places(table, toronto_grouped, 'Pharmacy')
x = np.arange(6)
ys = [i+x+(i*x)**2 for i in range(6)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



    # add markers to the map
markers_colors = []
for lat, lon, poi ,frq in zip(places['Latitude'],
                            places['Longitude'],
                            places['Neighborhood'],
                            frequncy):
    label = folium.Popup(str(poi) + "Category: {}, frequncy: {}".format('Pharmacy',frq) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[5],
        fill=True,
        fill_color=rainbow[5],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters  
#return map_clusters
def find_best_penalty_places(data,grouped_data,category,top_count = 10,top_places = 5,penalty =1):
    category_top = grouped_data.sort_values(by = category, axis=0, ascending=False).drop([category],axis = 1)[:top_count]
    def compare_distance_penalty(neighborhood,top = category_top,category = category, penalty = penalty):
        result = []
        coefs = []
        for i in range(10):
            result.append(cosine_similarity_1(neighborhood.drop(labels=category).
                             values[1:],top.iloc[i,1:].values))
            coefs.append(neighborhood[category])
        coefs = np.array([coefs])*sorted(result, reverse=True)[0]
        return np.mean(np.array(result)-coefs*penalty)
    top_places_found = grouped_data.T.apply(compare_distance_penalty)
    top5neighbors = grouped_data.iloc[top_places_found.sort_values(ascending = False)[:5].index,:]['Neighborhood'].values
    frequncy = grouped_data.iloc[top_places_found.sort_values(ascending = False)[:5].index,:][category]
    
    places = data[data['Neighborhood'].isin(top5neighbors)]
    return top5neighbors, places, frequncy
%%time
top5neighbors, places, frequncy = find_best_penalty_places(table, toronto_grouped, 'Pharmacy',penalty =1)
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



    # add markers to the map
markers_colors = []
for lat, lon, poi ,frq in zip(places['Latitude'],
                            places['Longitude'],
                            places['Neighborhood'],
                            frequncy):
    label = folium.Popup(str(poi) + "Category: {}, frequncy: {}".format('Pharmacy',frq) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[0],
        fill=True,
        fill_color=rainbow[0],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters  
#return map_clusters
Pharmacy_non_zero = toronto_grouped[toronto_grouped['Pharmacy'] !=0].sort_values(by = 'Pharmacy', axis=0, ascending=False)#.drop(['Pharmacy'],axis = 1)#, axis=0, ascending=False).drop(['Pharmacy'],axis = 1)[:10]
Pharmacy_non_zero.head()
Pharmacy_non_zero.shape
Pharmacy_zero = toronto_grouped[toronto_grouped['Pharmacy'] ==0]
Pharmacy_zero.shape
def compare_distance_new(neighborhood, top = Pharmacy_non_zero.reset_index(drop = True), category = 'Pharmacy'):
    result ={}
    for i in range(Pharmacy_non_zero.shape[0]):
        #We find similarity between neighborhoods which has target category neighborhoods without it
        result[top['Neighborhood'][i]] = cosine_similarity_1(neighborhood.drop(labels=category).iloc[1:].
                                                                     values,
                                                                     top.drop([category],axis = 1).iloc[i,1:].values)
    #Sort similarity values
    sorted_by_value = pd.Series(result).sort_values(ascending = False)
    sorted_by_freq = pd.Series(result).sort_values(ascending = False).index
    #Find avarage value for similarity of top 5 most similar neighborhoods
    return np.array([np.mean(sorted_by_value.values[:5]), np.mean(top[top['Neighborhood'].isin(sorted_by_freq[:5])][category])])
res = pd.DataFrame(np.array(Pharmacy_zero.T.apply(compare_distance_new)).T,columns = ['Similarity', 'Frequency'])
res.shape
res.sort_values(by = 'Similarity',ascending = False).iloc[:5,0]
def find_best_new_places(data,grouped_data,category,top_count = 10,top_places = 5):
    Pharmacy_non_zero = toronto_grouped[toronto_grouped['Pharmacy'] !=0].sort_values(by = 'Pharmacy', axis=0, ascending=False)
    Pharmacy_zero = toronto_grouped[toronto_grouped['Pharmacy'] ==0]
    def compare_distance_new(neighborhood, top = Pharmacy_non_zero.reset_index(drop = True), category = 'Pharmacy'):
        result ={}
        for i in range(Pharmacy_non_zero.shape[0]):
            result[top['Neighborhood'][i]] = cosine_similarity_1(neighborhood.drop(labels=category).iloc[1:].
                                                                     values,
                                                                     top.drop([category],axis = 1).iloc[i,1:].values)
        sorted_by_value = pd.Series(result).sort_values(ascending = False)
        sorted_by_freq = pd.Series(result).sort_values(ascending = False).index
        return np.array([np.mean(sorted_by_value.values[:5]), np.mean(top[top['Neighborhood'].isin(sorted_by_freq[:5])][category])])
    res = pd.DataFrame(np.array(Pharmacy_zero.T.apply(compare_distance_new)).T,columns = ['Similarity', 'Frequency'])
    top5_similar = res.sort_values(by = 'Similarity',ascending = False)[:5]
    top5neighbors = grouped_data.iloc[top5_similar[:5].index,:]['Neighborhood'].values
    frequncy = top5_similar.iloc[:,1]
    places = data[data['Neighborhood'].isin(top5neighbors)]
    return top5neighbors, places, frequncy
top5neighbors, places, frequncy = find_best_new_places(table, toronto_grouped, 'Pharmacy')
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



    # add markers to the map
markers_colors = []
for lat, lon, poi ,frq in zip(places['Latitude'],
                            places['Longitude'],
                            places['Neighborhood'],
                            frequncy):
    label = folium.Popup(str(poi) + "Category: {}, frequncy: {}".format('Pharmacy',frq) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[5],
        fill=True,
        fill_color=rainbow[5],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters  
#return map_clusters
users = toronto_grouped.T
users.columns = toronto_grouped['Neighborhood']
users.drop(['Neighborhood']).head()
def cosine_similarity_3(v1, v2 =users.loc[['Pharmacy'],:].values[0] ):
    result = 0
    prod = np.dot(v1, v2)
    len1 = math.sqrt(np.dot(v1, v1))
    len2 = math.sqrt(np.dot(v2, v2))
    try:
        result = prod / (len1 * len2)
    except:
        return result
    return result
top5_venues = users.drop(['Neighborhood']).apply(cosine_similarity_3,axis = 1).sort_values(ascending = False)[1:6]
top5_venues
users.loc[top5_venues.index,:].sum().sort_values(ascending = False)[:5]/5
top5neighbors = users.loc[top5_venues.index,:].sum().sort_values(ascending = False).index[:5]
top5neighbors
frequency = users.loc[top5_venues.index,:].sum().sort_values(ascending = False).values[:5]/5
frequency
def collaborate_filter(data,grouped_data,category):
    venues = grouped_data.T
    venues.columns = grouped_data['Neighborhood']
    def cosine_similarity_3(v1, v2 =venues.loc[[category],:].values[0] ):
        result = 0
        prod = np.dot(v1, v2)
        len1 = math.sqrt(np.dot(v1, v1))
        len2 = math.sqrt(np.dot(v2, v2))
        try:
            result = prod / (len1 * len2)
        except:
            return result
        return result
    #find top 5 similarity venues. [1:6] - because start from simalarity = 1 for category = category
    top5_venues = venues.drop(['Neighborhood']).apply(cosine_similarity_3,axis = 1).sort_values(ascending = False)[1:6]
    #find most similar venues
    top5neighbors = venues.loc[top5_venues.index,:].sum().sort_values(ascending = False).index[:5]
    #find average  frequencies for each Neighborhood in top 5 venues
    frequency = venues.loc[top5_venues.index,:].sum().sort_values(ascending = False).values[:5]/5
    #retun only top 5 sorted by frquency values
    places = data[data['Neighborhood'].isin(top5neighbors)]
    return top5neighbors, places, frequency
top5neighbors, places, frequncy = collaborate_filter(table, toronto_grouped, 'Pharmacy')
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



    # add markers to the map
markers_colors = []
for lat, lon, poi ,frq in zip(places['Latitude'],
                            places['Longitude'],
                            places['Neighborhood'],
                            frequncy):
    label = folium.Popup(str(poi) + "Category: {}, frequncy: {}".format('Pharmacy',frq) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[1],
        fill=True,
        fill_color=rainbow[1],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters  
#return map_clusters

manhattan_data = pd.read_csv('../input/ibm-capstone-project-geodata/manhattan_data.csv')
manhattan_venues = pd.read_csv('../input/ibm-capstone-project-geodata/manhattan_venues.csv')
manhattan_venues.head()
manhattan_venues.groupby('Neighborhood').count()[:5]
print('There are {} uniques categories.'.format(len(manhattan_venues['Venue Category'].unique())))
# one hot encoding
manhattan_onehot = pd.get_dummies(manhattan_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
manhattan_onehot['Neighborhood'] = manhattan_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])
manhattan_onehot = manhattan_onehot[fixed_columns]

manhattan_onehot.head()
manhattan_onehot.shape
manhattan_grouped = manhattan_onehot.groupby('Neighborhood').mean().reset_index()
manhattan_grouped.head()
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
neighborhoods_venues_sorted['Neighborhood'] = manhattan_grouped['Neighborhood']

for ind in np.arange(manhattan_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(manhattan_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head(10)
top5neighbors, places, frequncy = collaborate_filter(manhattan_data, manhattan_grouped, 'Gym / Fitness Center')
places
address = 'Manhattan, NY'

#geolocator = Nominatim()
#location = geolocator.geocode(address)
latitude = 40.7900869
longitude = -73.9598295
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)



    # add markers to the map
markers_colors = []
for lat, lon, poi ,frq in zip(places['Latitude'],
                            places['Longitude'],
                            places['Neighborhood'],
                            frequncy):
    label = folium.Popup(str(poi) + "Category: {}, frequncy: {}".format('Pharmacy',frq) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[0],
        fill=True,
        fill_color=rainbow[0],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters  
#return map_clusters
