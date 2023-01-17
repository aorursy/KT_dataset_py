import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

import requests

from geopy.geocoders import Nominatim

import folium

import re

import json

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules

import matplotlib.cm as cm



import matplotlib.colors as colors
# address = 'New Delhi'

# geolocator = Nominatim(user_agent="ny_explorer")

# location = geolocator.geocode(address)

# latitude = location.latitude

# longitude = location.longitude

# print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
latitude = 28.6141793

longitude = 77.2022662
#accessing the web page by Http request made by requests library

# req = requests.get("https://en.wikipedia.org/wiki/Neighbourhoods_of_Delhi").text

# soup = BeautifulSoup(req, 'lxml')

# div = soup.find('div', class_="mw-parser-output" )

# print("web Page Imported")

# #Code to extract the relevent data from the request object using beautiful soup

# data = pd.DataFrame(columns=['Borough','Neighborhood'])

# i=-1

# flag = False

# no=0

# prev_borough = None

# for child in div.children:

#     if child.name:

#         span = child.find('span')

#         if span!=-1 and span is not None:

#             try:

#                 if span['class'][0] == 'mw-headline' and child.a.text!='edit':

#                     prev_borough = child.a.text

#                     i+=1

#                     flag = True

#                     continue

#             except KeyError:

#                 continue

#         if child.name=='ul' and flag==True:

#             neighborhood = []

#             for ch in child.children:

                

#                 try:

#                     data.loc[no]=[prev_borough,ch.text]

#                     no+=1

#                 except AttributeError:

#                     pass

#         flag = False

# data[50:60]
# lat_lng = pd.DataFrame(columns=['latitude','longitude'])

# geolocator = Nominatim(user_agent="ny_explorer")

# for i in range(184):

#     address = data['Neighborhood'].loc[i]+',New Delhi'

#     try: 

#         location = geolocator.geocode(address)

#         lat_lng.loc[i]=[location.latitude,location.longitude]

#     except AttributeError:

#         continue
# df1 = data

# df2 = lat_lng

# delhi_neighbourhood_data = pd.concat([df1, df2], axis=1)

# delhi_neighbourhood_data.to_csv(r'E:\jupyter\Coursera Practice\delhi_dataSet.csv')
delhi_neighborhood_data = pd.read_csv(r'../input/delhi_dataSet.csv')

delhi_neighborhood_data.dropna(inplace=True)

delhi_neighborhood_data.reset_index(inplace=True)

delhi_neighborhood_data.drop(['index','Unnamed: 0'], axis=1, inplace=True)

delhi_neighborhood_data.head()
delhiData = delhi_neighborhood_data

map_delhi = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to map

for lat, lng, label in zip(delhiData['latitude'], delhiData['longitude'], delhiData['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_delhi)  

    

map_delhi
# CLIENT_ID = 'PVEHZMCGQRW1UTUDAKHLC0RTRNC205YZ2NJDZDPPJOHQV5VH' # your Foursquare ID

# CLIENT_SECRET = 'XYAYEPCDCHKUT44EMD25OADY1UADBPQZEGVYH0IJRDEWKW1Q' # your Foursquare Secret

# VERSION = '20180605' # Foursquare API version

# radius = 1000

# LIMIT = 200



# print('Credentails Registered')
# def getNearbyVenues(names, latitudes, longitudes, radius=500):

    

#     venues_list=[]

#     for name, lat, lng in zip(names, latitudes, longitudes):

#         print(name)

            

#         # create the API request URL

#         url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&categoryId={}&ll={},{}&radius={}&limit={}'.format(

#             CLIENT_ID, 

#             CLIENT_SECRET, 

#             VERSION,

#             '4d4b7105d754a06374d81259',

#             lat, 

#             lng, 

#             radius, 

#             LIMIT)

            

#         # make the GET request

#         try:

#             results = requests.get(url).json()["response"]['groups'][0]['items']

#         except KeyError:

#             continue

        

#         # return only relevant information for each nearby venue

#         venues_list.append([(

#             name, 

#             lat, 

#             lng, 

#             v['venue']['name'], 

#             v['venue']['location']['lat'], 

#             v['venue']['location']['lng'],  

#             v['venue']['categories'][0]['name']) for v in results])



#     nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

#     nearby_venues.columns = ['Neighborhood', 

#                   'Neighborhood Latitude', 

#                   'Neighborhood Longitude', 

#                   'Venue', 

#                   'Venue Latitude', 

#                   'Venue Longitude', 

#                   'Venue Category']

    

#     return(nearby_venues)
# delhi_venues = getNearbyVenues(names=delhiData['Neighborhood'],

#                                    latitudes=delhiData['latitude'],

#                                    longitudes=delhiData['longitude']

#                                   )
delhi_venues = pd.read_csv(r'../input/restaurant_dataSet.csv')
map_res = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to map

for lat, lng, label in zip(delhi_venues['Venue Latitude'], delhi_venues['Venue Longitude'], delhi_venues['Venue']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=2,

        popup=label,

        color='red',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_res)  

    

map_res
# one hot encoding

delhi_onehot = pd.get_dummies(delhi_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe

delhi_onehot['Neighborhood'] = delhi_venues['Neighborhood'] 



# move neighborhood column to the first column

fixed_columns = [delhi_onehot.columns[-1]] + list(delhi_onehot.columns[:-1])

delhi_onehot = delhi_onehot[fixed_columns]



delhi_onehot.head()
delhi_onehot.shape
#To be used while Generating Graphs

delhi_grouped = delhi_onehot.groupby('Neighborhood').mean().reset_index()

delhi_grouped.head()
for i in delhi_grouped.columns:

    print(i,end=", ")
delhi_grouped.shape
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

neighborhoods_venues_sorted['Neighborhood'] = delhi_grouped['Neighborhood']



for ind in np.arange(delhi_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(delhi_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted.head()
# set number of clusters

kclusters = 5



delhi_grouped_clustering = delhi_grouped.drop('Neighborhood', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(delhi_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10]
# add clustering labels

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)



delhi_merged = delhiData



# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

delhi_merged = delhi_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')



delhi_merged.dropna(inplace=True)

delhi_merged.head() # check the last columns!
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(delhi_merged['latitude'], delhi_merged['longitude'], delhi_merged['Neighborhood'], delhi_merged['Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[int(cluster)-1],

        fill=True,

        fill_color=rainbow[int(cluster)-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters
clusterdata = pd.merge(delhi_onehot.groupby('Neighborhood').sum(),delhi_merged[['Neighborhood','Cluster Labels']],left_on='Neighborhood', right_on='Neighborhood',how='inner')

clusterdata = clusterdata.iloc[:,1:].groupby('Cluster Labels').sum().transpose()

clusterdata.head()
import seaborn as sns
def plot_bar(clusternumber):

    sns.set(style="whitegrid",rc={'figure.figsize':(20,10)})

    df = clusterdata[[clusternumber]].drop(clusterdata[[clusternumber]][clusterdata[clusternumber]==0].index)

    chart = sns.barplot(x=df.index, y=clusternumber, data=df)

    chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plot_bar(0)
plot_bar(1)
plot_bar(2)
plot_bar(3)
plot_bar(4)
delhi_venues.drop('Unnamed: 0',axis=1,inplace=True)
forheatmap=delhi_venues.copy()

forheatmap=pd.merge(forheatmap,delhi_merged[['Neighborhood','Cluster Labels']],left_on='Neighborhood', right_on='Neighborhood',how='inner')

forheatmap.drop(forheatmap[~forheatmap['Cluster Labels'].isin([1,2])].index, inplace=True)
forheatmap.head()
from folium.plugins import HeatMap
#heat map of all restaurants in selected Neighborhoods

res_heat = folium.Map(location=[latitude, longitude], zoom_start=11)

HeatMap(list(zip(forheatmap['Venue Latitude'],forheatmap['Venue Longitude'])),

        min_opacity=0.2,

        radius=10, blur=15,

        max_zoom=1

       ).add_to(res_heat)

for lat, lng, label in zip(forheatmap['Neighborhood Latitude'], forheatmap['Neighborhood Longitude'], forheatmap['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=2,

        popup=label,

        color='red',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(res_heat)

res_heat
forindres = forheatmap[forheatmap['Venue Category']=='Indian Restaurant']



# heat map for Indian Restaurants in the selected Neighborhoods

res_heat_ind = folium.Map(location=[latitude, longitude], zoom_start=11)

HeatMap(list(zip(forindres['Venue Latitude'],forindres['Venue Longitude'])),

        min_opacity=0.2,

        radius=10, blur=15,

        max_zoom=1

       ).add_to(res_heat_ind)

for lat, lng, label in zip(forindres['Neighborhood Latitude'], forindres['Neighborhood Longitude'], forindres['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=2,

        popup=label,

        color='red',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(res_heat_ind)

res_heat_ind
count_all = forheatmap[['Neighborhood','Venue']].groupby('Neighborhood').count().sort_values(by='Venue')

target_count = int(0.6*len(count_all))

print(count_all.iloc[target_count])

count_all.drop(count_all[count_all.Venue.values>7].index,inplace=True)

count_all.columns=['all count']

count_all.head()
count_ind = forheatmap[forheatmap['Venue Category']=="Indian Restaurant"][['Neighborhood','Venue']].groupby('Neighborhood').count().sort_values(by='Venue')

target_count = int(0.3*len(count_ind))

print(count_ind.iloc[target_count])

count_ind.drop(count_ind[count_ind.Venue.values>1].index,inplace=True)

count_ind.columns = ['ind count']

count_ind.head()
lowdensity = count_all.join(count_ind)

lowdensity.index.values
temp_recommend = delhiData.copy()

temp_recommend.drop(temp_recommend[~temp_recommend['Neighborhood'].isin(lowdensity.index.values)].index, inplace=True)

temp_recommend.head()
#most popular neighborhoods

top_nei = delhi_venues[['Neighborhood','Venue']].groupby('Neighborhood').count().sort_values(by='Venue', ascending=False).head(3).index.values

top_nei
toplatlng = delhiData[['Neighborhood','latitude','longitude']][delhiData['Neighborhood'].isin(top_nei)].reset_index()

toplatlng
from math import sin, cos, sqrt, atan2, radians



def distanceInKM(la1,lo1,la2,lo2):

    # approximate radius of earth in km

    R = 6373.0

    

    lat1 = radians(la1)

    lon1 = radians(lo1)

    lat2 = radians(la2)

    lon2 = radians(lo2)



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))



    dis = R * c

    return round(dis,4)



print("Result:", distanceInKM(toplatlng.iloc[2]['latitude'],toplatlng.iloc[2]['longitude'],toplatlng.iloc[0]['latitude'],toplatlng.iloc[0]['longitude']))
temp_recommend.reset_index(inplace=True)
temp_recommend.drop(columns=['index','Borough'], inplace=True)
temp_recommend.head()
for i in toplatlng.index:

    temp_recommend[toplatlng.iloc[i]['Neighborhood']] = temp_recommend.apply(lambda x : distanceInKM(toplatlng.iloc[i]['latitude'],toplatlng.iloc[i]['longitude'],x['latitude'],x['longitude']),axis=1)
temp_recommend.head()
# top 5 neighborhoods near Connaught Place

neiNearCP = temp_recommend.sort_values(by=['Connaught Place']).iloc[:,:3].head().set_index('Neighborhood')

neiNearCP
# top 5 neighborhoods near Hauz Khas Village

neiNearHK = temp_recommend.sort_values(by=['Hauz Khas Village']).iloc[:,:3].head().set_index('Neighborhood')

neiNearHK
# top 5 neighborhoods near Khirki Village

neiNearKV = temp_recommend.sort_values(by=['Khirki Village']).iloc[:,:3].head().set_index('Neighborhood')

neiNearKV
final_recommend=neiNearCP.append(neiNearHK).append(neiNearKV).reset_index()

final_recommend.drop_duplicates(inplace=True)

final_recommend.reset_index(inplace=True)

final_recommend.drop(columns=['index'],inplace=True)

final_recommend
final = folium.Map(location=[latitude, longitude], zoom_start=11)



# add markers to map

for lat, lng, label in zip(final_recommend['latitude'], final_recommend['longitude'], final_recommend['Neighborhood']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(final)  

    

final
import pandas as pd

delhi_dataSet = pd.read_csv("../input/delhi_dataSet.csv")

restaurant_dataSet = pd.read_csv("../input/restaurant_dataSet.csv")