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
# Read the file and create a pandas df with it



import pandas as pd

df_cdmx = pd.read_csv('/kaggle/input/mx-neighborhoods-geo-rent/MX_Neighborhoods_Geo_Rent.csv')

df_cdmx.head()
# Filter municipalities

mask = (df_cdmx['Municipality'] == 'CUAUHTEMOC') | (df_cdmx['Municipality'] == 'MIGUEL HIDALGO')

df_cdmx.loc[mask]

df_cdmx = df_cdmx.loc[mask]
df_cdmx
# Install required components to handle the data and analyse it

import itertools

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as colors

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
# Create dataframe to plot the average rent by neighborhood

df_cdmx_rent = df_cdmx.groupby('Neighborhood').mean().reset_index()

df_cdmx_rent.drop(columns=['Latitude', 'Longitude'], inplace=True)

df_cdmx_rent.sort_values(by=['Rent'], ascending=False, inplace=True)

df_cdmx_rent.set_index('Neighborhood', inplace=True)

df_cdmx_rent
# Calculate basic statistics



df_cdmx_rent.describe()
# Create boxplot



df_cdmx_rent.plot(kind='box', figsize=(5,8), color='blue')



plt.title('Average rent distribution') # add title to the plot



plt.show()
# Create histogram



df_cdmx_rent.plot(kind='hist', figsize=(10,5), color='blue')



plt.title('Average rent distribution') # add title to the plot



plt.show()
# Create bar graph



df_cdmx_rent.plot(kind='bar', figsize=(30, 15), )



plt.xlabel('Neighborhood') # add to x-label to the plot

plt.ylabel('Average rent') # add y-label to the plot

plt.title('Average Rent by Neighborhood') # add title to the plot



plt.show()
# Install Mominatim to get the geolocation information of User's office in Mexico City



!conda install -c conda-forge geopy --yes 

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
# Get Folium to create the map



!conda install -c conda-forge folium=0.5.0 --yes

import folium # map rendering library



print('Libraries imported.')
# Generate Mexico City's lat and lon to create a map in the next step



address = 'Mexico City'



geolocator = Nominatim(user_agent="mx_explorer")

cdmx_location = geolocator.geocode(address)

cdmx_latitude = cdmx_location.latitude

cdmx_longitude = cdmx_location.longitude

print('The geograpical coordinate of Mexico City are {}, {}.'.format(cdmx_latitude, cdmx_longitude))
# create map of Mexico city showing the tz location and all neighborhoods



cdmx = folium.Map(location=[cdmx_latitude, cdmx_longitude], zoom_start=12)

 

# add markers to map

for lat, lng, borough, neighborhood, average_rent in zip(df_cdmx['Latitude'], df_cdmx['Longitude'], df_cdmx['Municipality'], df_cdmx['Neighborhood'], df_cdmx['Rent']):

    label = 'Avg Rent $ {}, Bh {}, Nbh {}'.format(average_rent, borough, neighborhood)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='Purple',

        fill=True,

        fill_color='#9900FF',

        fill_opacity=0.7,

        parse_html=False).add_to(cdmx) 

    

cdmx
# Install components required to handle JSON files



import json # library to handle JSON files

import requests # library to handle requests

from pandas import json_normalize # tranform JSON file into a pandas dataframe
# Save variables required to access Foursquare: CLIENT_ID, CLIENT_SECRET and VERSION



CLIENT_ID = 'ZRDR4VMLR3U14PHLUME3QC1CG5L4XXUBSKE4COCHEK4VBWJX' # your Foursquare ID

CLIENT_SECRET = 'ITU1FS52ILD04RTHIZT1VE2JBYYUTQ25BQ14FY5IYHVEW0FN' # your Foursquare Secret

VERSION = '20200601' # Foursquare API version
# Function to get near venues for all neighborhoods







LIMIT = 100



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
# Create dataframe with venues information



cdmx_venues = getNearbyVenues(names=df_cdmx['Neighborhood'],

                                   latitudes=df_cdmx['Latitude'],

                                   longitudes=df_cdmx['Longitude']

                                  )
# Check dataframe with venues information



print(cdmx_venues.shape)

cdmx_venues.head(10)
# Filter venue category to keep only Convinience stores



stores = cdmx_venues.loc[cdmx_venues['Venue Category'].isin(['Convenience Store'])]
stores
# Merge dataframes



df_merged = pd.merge(stores, df_cdmx, how='left', on='Neighborhood', left_on=None, right_on=None,

         left_index=False, right_index=False, sort=True,

         suffixes=('_x', '_y'), copy=True, indicator=False,

         validate=None)



df_merged
#Drop columns not to be used as variables



df_merged.drop(columns=['Neighborhood_Municipality', 'Latitude', 'Longitude'], inplace=True)
df_merged
# Remove duplicates

df_merged.drop_duplicates(subset=['Venue Latitude', 'Venue Longitude'], keep='first', inplace=True)
df_merged
# create map of Mexico city showing the tz location and all venues



stores = folium.Map(location=[cdmx_latitude, cdmx_longitude], zoom_start=12)

 

# add markers to map

for lat, lng, borough, neighborhood, venue in zip(df_merged['Venue Latitude'], df_merged['Venue Longitude'], df_merged['Municipality'], df_merged['Neighborhood'], df_merged['Venue']):

    label = 'Venue {}, Bh {}, Nbh {}'.format(venue, borough, neighborhood)

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='Red',

        fill=True,

        fill_color='#9900FF',

        fill_opacity=0.7,

        parse_html=False).add_to(stores) 

    

stores
df_merged.to_csv('list of stores.csv') # Save dataframe as CSV
# Read the file containing the Social Development Index per neighbourhood



dev_index = pd.read_csv('/kaggle/input/cdmx-soc-dev-index/MX_Neighborhoods_Social_Dev_Index.csv')

dev_index.head()
dev_ind_plot = dev_index.groupby('Municipality')['Soc_Dev_Ind'].mean() # Create a dataframe with the average SDI per municipality
dev_ind_plot
# Create bar graph



dev_ind_plot.plot(kind='bar')



plt.xlabel('Municipality') # add to x-label to the plot

plt.title('Social Development Index') # add title to the plot



plt.show()
dev_ind_neigh = dev_index.groupby('Neighborhood')['Soc_Dev_Ind'].mean() # Create a dataframe with the average SDI per neighbourhood
dev_ind_neigh
dev_ind_neigh.sort_values(ascending=False).plot.bar(figsize=(30,10)) # Plot SDI per neighbourhood
# Create SDI histogram



dev_ind_neigh.plot(kind='hist')



plt.title('SDI by Neighborhood') # add title to the plot



plt.show()
# Merge SDI dataframe



df_final = pd.merge(df_merged, dev_index, how='left', on='Neighborhood', left_on=None, right_on=None,

         left_index=False, right_index=False, sort=True,

         suffixes=('_x', '_y'), copy=True, indicator=False,

         validate=None)



df_final
df_final.drop(columns=['Municipality_y'], inplace=True) # Drop duplicated columns
df_final['Soc_Dev_Ind'] = df_final['Soc_Dev_Ind'].fillna(0) # Fill NaN with zeros
df_final
df_final.rename(columns = {'Municipality_x': 'Municipality'}, inplace = True) # Rename municipality column
df_final
# Apply Levenshtein function to uniform venue description names



import Levenshtein



df_final['Venue'] = df_final['Venue'].str.lower().str.strip() # normalize names

df_final['distance'] = df_final.apply(lambda x: Levenshtein.distance(x['Venue'],

                                               df_final.groupby(['Venue'])['Venue'].\

                                                               value_counts().\

                                                               idxmax()[0]),axis=1)

df_final['converted'] = df_final.apply(lambda x: x['Venue'].strip() 

                                     if x['distance'] <= 5 

                                     else np.nan,axis=1)
df_final
df_final.to_csv('df_final.csv') # Save dataframe to CSV
# Standarize venue names



df_final["Venue"]= df_final["Venue"].str.replace("7 eleven", "7- eleven", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("7-eleven", "7- eleven", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("cÃ­rculo k", "circle k", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("super k", "circle k", case = False) 

df_final["Venue"]= df_final["Venue"].str.replace("círculo k", "circle k", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("mini bodega aurrera", "bodega aurrera", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("bodega aurrera express", "bodega aurrera", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("oxxo canal del norte", "oxxo", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("oxxo cholula y benjamin hill", "oxxo", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("oxxo constituyentea", "oxxo", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("oxxo poniente 44", "oxxo", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("tiendas 3b plomo", "tiendas 3b", case = False)
df_final
# Last venue names to standarize



df_final["Venue"]= df_final["Venue"].str.replace("bodega aurrera expres", "bodega aurrera", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("bodega aurrerÃ¡ express", "bodega aurrera", case = False)

df_final["Venue"]= df_final["Venue"].str.replace("bodega aurrerá express", "bodega aurrera", case = False)
df_final.drop(columns=['converted', 'distance'], inplace=True) # Remove unnecessary columns
df_final = df_final[df_final['Venue'] != 'walmart toreo'] # Drop walmarts since belong to a different category of stores

df_final = df_final[df_final['Venue'] != 'walmart buenavista']
df_final
df_one_hot = df_final.copy() # Copy dataframe to apply one-hot encoding
# Drop columns not used as variables

df_one_hot.drop(columns=['Neighborhood', 'Venue Category', 'Municipality', 'Neighborhood_Municipality'], inplace=True) 
df_one_hot
# one hot encoding

df_cluster = pd.get_dummies(df_one_hot, columns=['Venue'], prefix="", prefix_sep="")
df_cluster
from sklearn.cluster import KMeans # Import K-Means algorithm
# Normalize variables before applying K-Means



from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler().fit(df_cluster)

df_cluster_tr = norm.transform(df_cluster)
df_cluster_tr
# set number of clusters

kclusters = 6



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=42).fit(df_cluster_tr)
# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:67]
# add clustering labels

df_cluster.insert(0, 'Cluster Labels', kmeans.labels_)
df_cluster
# Add back Venue and neighbourhood to dataframe



df_cluster['Venue']=df_final['Venue']

df_cluster['Neighborhood']=df_final['Neighborhood']
df_cluster
df_cluster.to_csv('df_cluster.csv') # Save dataframe as CSV
# create map

map_clusters = folium.Map(location=[cdmx_latitude, cdmx_longitude], zoom_start=12)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, nbh, venue, cluster in zip(df_cluster['Venue Latitude'], df_cluster['Venue Longitude'], df_cluster['Neighborhood'], df_cluster['Venue'], df_cluster['Cluster Labels']):

    label = folium.Popup(str(venue) + ',' + str(nbh) + ',' + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters
# Group DF by clusters to determine rent and sdi average per cluster

df_cluster_anal = df_cluster.groupby('Cluster Labels')['Rent', 'Soc_Dev_Ind'].mean()
df_cluster_anal
df_cluster_anal.sort_values(by=['Rent']) # Sort df by rent