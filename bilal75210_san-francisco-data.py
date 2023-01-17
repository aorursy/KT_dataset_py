# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



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



#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab

import folium # map rendering library



print('Libraries imported.')
df = pd.read_csv('/kaggle/input/sf-parks/SF_Park_Scores.csv')

df.head()
df.info()
df.isnull().sum(axis = 0)
df = df.rename(columns={'Facility Type':'FacilityType','Square Feet':'SquareFeet','Perimeter Length':'PerimeterLength'})
df.tail()
park_data = df
all_data_na = (park_data.isnull().sum() / len(park_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
#Data Cleaning Process

# I will drop the Floor Count    

park_data = park_data.drop(["Floor Count"], axis = 1)

park_data = park_data.dropna()

#I will put 0 for numeric data

park_data["Latitude"] = park_data["Latitude"].fillna(0)

park_data["Longitude"] = park_data["Longitude"].fillna(0)

park_data["Acres"] = park_data["Acres"].fillna(0)

park_data["Perimeter Length"] = park_data["PerimeterLength"].fillna(0)

park_data["Square Feet"] = park_data["SquareFeet"].fillna(0)

park_data["Zipcode"] = park_data["Zipcode"].fillna(0)

#I will put None for strings

park_data["State"] = park_data["State"].fillna("None")

park_data["Address"] = park_data["Address"].fillna("None")

park_data["Facility Name"] = park_data["Facility Name"].fillna("None")

park_data["FacilityType"] = park_data["FacilityType"].fillna("None")
#info about the dataset

park_data.info()
park_data.Zipcode.unique()


#the distrubution of parks in san francisco in terms of longtitude and latitude

longitude = list(park_data.Longitude) 

latitude = list(park_data.Latitude)

plt.figure(figsize = (10,10))

plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 30)

plt.show()
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values



# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors



# import k-means from clustering stage

from sklearn.cluster import KMeans



#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab

import folium # map rendering library



print('Libraries imported.')
address = 'san francisco'



geolocator = Nominatim(user_agent="san francisco")

location = geolocator.geocode(address)

latitude_toronto = location.latitude

longitude_toronto = location.longitude

print('The geograpical coordinate of san francisco are {}, {}.'.format(latitude_toronto, longitude_toronto))
map_toronto = folium.Map(location=[latitude_toronto, longitude_toronto], zoom_start=10)



# add markers to map

for lat, lng, borough, Neighbourhood in zip(park_data['Latitude'], park_data['Longitude'], park_data['Park'], park_data['Score']):

    label = '{}, {}'.format(Neighbourhood, borough)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_toronto)  

    

map_toronto
plt.subplots(figsize=(22,12))

sns.countplot(y=park_data['Zipcode'],order=park_data['Zipcode'].value_counts().index)

plt.show()
import matplotlib as mpl

from wordcloud import WordCloud, STOPWORDS



mpl.rcParams['font.size']=12                

mpl.rcParams['savefig.dpi']=100             

mpl.rcParams['figure.subplot.bottom']=.1 

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(park_data['Park']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=1000)
#SF Parks Facility Score points

park_data['Score'].value_counts().sort_index().plot.line(figsize=(12, 6),color='mediumvioletred',fontsize=16,title='SF Parks Score')
# Which Public Administation has the highest parks 

import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True) # this is important



z = {'PSA1': 'PSA1', 'PSA2': 'PSA2', 'PSA3': 'PSA3','PSA4': 'PSA4','PSA5': 'PSA5','PSA6': 'PSA6','GGP': 'GGP'}

data = [go.Bar(

            x = park_data.PSA.map(z).unique(),

            y = park_data.PSA.value_counts().values,

            marker= dict(colorscale='Jet',

                         color = park_data.PSA.value_counts().values

                        ),

           

    )]



layout = go.Layout(

    title='Target variable distribution'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
def getNearbyVenues(names, latitudes, longitudes, radius=500):

    

    venues_list=[]

    for name, lat, lng in zip(names, latitudes, longitudes):

        print(name)

            

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
CLIENT_ID = 'MIF2SPKPEZP0YIUKZMPBLCA3R4ESWCFJDYPFTPHV4PTFVXIO' # your Foursquare ID

CLIENT_SECRET = '5UOPEX5O43CRW0TZWYYOXST2VJBTPKIXAJWJ2TJVH3S23ZEK' # your Foursquare Secret

VERSION = '20180604'

LIMIT = 30

print('Your credentails:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
manhattan_venues = getNearbyVenues(names=park_data['Park'].head(20),

                                   latitudes=park_data['Latitude'],

                                   longitudes=park_data['Longitude']

                                  )
print(manhattan_venues.shape)

manhattan_venues.head()
manhattan_venues.groupby('Neighborhood').count()
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

manhattan_grouped
manhattan_grouped.shape
num_top_venues = 5



for hood in manhattan_grouped['Neighborhood']:

    print("----"+hood+"----")

    temp = manhattan_grouped[manhattan_grouped['Neighborhood'] == hood].T.reset_index()

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

neighborhoods_venues_sorted['Neighborhood'] = manhattan_grouped['Neighborhood']



for ind in np.arange(manhattan_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(manhattan_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted.head()
# set number of clusters

kclusters = 5



manhattan_grouped_clustering = manhattan_grouped.drop('Neighborhood', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(manhattan_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10] 
# add clustering labels

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

#park_data.drop(['Cluster Labels'], axis=1)

manhattan_merged = park_data



# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

manhattan_merged = manhattan_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Park')



manhattan_merged.head() # check the last columns!
address = 'san francisco'



geolocator = Nominatim(user_agent="san francisco")

location = geolocator.geocode(address)

latitude_toronto = location.latitude

longitude_toronto = location.longitude

print('The geograpical coordinate of san francisco are {}, {}.'.format(latitude_toronto, longitude_toronto))
map_clusters = folium.Map(location=[latitude_toronto, longitude_toronto], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(manhattan_merged['Latitude'], manhattan_merged['Longitude'], manhattan_merged['Park'], manhattan_merged['Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        #color=rainbow[kcluster-1],

        fill=True,

        #fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 0, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 1, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 2, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 3, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 4, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]