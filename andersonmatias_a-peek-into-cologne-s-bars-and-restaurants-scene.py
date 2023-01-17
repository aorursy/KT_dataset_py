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
# Import all the required libraries and modules

import requests # library to handle requests

import pandas as pd # library for data analsysis

import numpy as np # library to handle data in a vectorized manner

import random # library for random number generation



!conda install -c anaconda lxml --yes

!conda install -c anaconda beautifulsoup4 --yes

from bs4 import BeautifulSoup



# libraries for displaying images

from IPython.display import Image 

from IPython.core.display import HTML 



# library for chart plotting

import matplotlib as mpl

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler # Module from Scikit-learn for Data normalization

from sklearn.cluster import KMeans # Module from Scikit-learn for clustering

from sklearn.datasets.samples_generator import make_blobs

!wget --quiet https://cocl.us/Geospatial_data

!conda install -c conda-forge folium=0.5.0 --yes

import folium

    

# tranforming json file into a pandas dataframe library

from pandas.io.json import json_normalize



print('Folium installed')

print('Libraries imported.')
cgn_df = pd.read_html('https://en.wikipedia.org/wiki/Districts_of_Cologne')[1]

cgn_df = cgn_df.drop([9,10])

cgn_df.drop(['Map','Coat','City parts','District Councils','Town Hall'], axis=1, inplace =True)

cgn_df.rename(columns={'City district':'District','Population1':'Population','Pop. density':'Pop_Density'}, inplace=True)

cgn_df['District'].replace({'District 1 Köln-Innenstadt':'Köln-Innenstadt','District 2 Köln-Rodenkirchen':'Köln-Rodenkirchen','District 3 Köln-Lindenthal':'Köln-Lindenthal','District 4 Köln-Ehrenfeld':'Köln-Ehrenfeld','District 5 Köln-Nippes':'Köln-Nippes','District 6 Köln-Chorweiler':'Köln-Chorweiler','District 7 Köln-Porz':'Köln-Porz','District 8 Köln-Kalk':'Köln-Kalk','District 9 Köln-Mülheim':'Köln-Mülheim'}, inplace=True)

cgn_df.sort_values('Pop_Density', inplace=True, ascending=False)

cgn_df['Pop_Density'] = cgn_df['Pop_Density'].str.replace('/km²','').str.replace('.','').astype(float)

cgn_df
labels = ['Köln-Innenstadt','Köln-Ehrenfeld','Köln-Nippes','Köln-Lindenthal','Köln-Kalk','Köln-Mülheim','Köln-Rodenkirchen','Köln-Porz','Köln-Chorweiler']



ind = np.arange(len(cgn_df['Pop_Density']))  

width = 0.3



fig, ax = plt.subplots(figsize=(16,8))

rects = ax.bar(ind, cgn_df['Pop_Density'], width, label=labels, color='#5bc0de')

ax.set_title("Cologne Neighborhoods - Population Density Bar Plot", fontsize=16)

ax.set_xticks(ind)

ax.set_xticklabels((labels))

plt.ylabel('Population Density (Pop./km²)')

plt.xlabel('Districts')

ax.get_yaxis().set_visible(True)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(True)

ax.spines['right'].set_visible(False)







def autolabel(rects, xpos='center'):



    ha = {'center': 'center', 'right': 'left', 'left': 'right'}

    offset = {'center': 0, 'right': 1, 'left': -1}



    for rect in rects:

        height = rect.get_height().round(2)

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(offset[xpos]*3, 3),  # use 3 points offset

                    textcoords="offset points",  # in both directions

                    ha=ha[xpos], va='bottom', fontsize=14)

        

autolabel(rects, "center")



fig.tight_layout()



plt.show()
cgn_df.drop(cgn_df.index[4:], axis=0, inplace=True)

cgn_df
!conda install -c conda-forge geopy --yes 

from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
geolocator = Nominatim(user_agent="Cologne Explorer")

cgn_df['Districts_Coord'] = cgn_df['District'].apply(geolocator.geocode).apply(lambda x: (x.latitude, x.longitude))

cgn_df[['Latitude', 'Longitude']] = cgn_df['Districts_Coord'].apply(pd.Series)

cgn_df.drop(['Districts_Coord'], axis=1, inplace=True)

cgn_df
rent_df = pd.read_html('https://www.koeln.de/immobilien/mietspiegel.html')[0]

rent_df.drop(['3. Quartal 2018','Veränderung'], axis=1, inplace=True)

rent_df.rename(columns={'Stadtteil':'District','4. Quartal 2018/04':'Price_m2_2018'}, inplace=True)

rent_df.head()
neigh_list = ['Innenstadt','Ehrenfeld','Nippes','Lindenthal']

rent_array = rent_df[rent_df.District.isin(neigh_list)]

rent_df2 = pd.DataFrame(rent_array, columns=['District','Price_m2_2018'])

rent_df2['Price_m2_2018'] = rent_df2['Price_m2_2018'].str.replace('€','').str.replace(',','.').astype(float)

rent_df2.sort_values('Price_m2_2018', inplace=True, ascending=False)

rent_df2
# No term "Innestadt" found on rent_df, therefore district was missing on rent_df2. Being the Innenstadt district mainly composed of the Altstadt-Süd und Altstadt-Nord, it was taken the average of both to be appended into the new dataframe

innenstadt = pd.DataFrame({'District':['Innenstadt'],'Price_m2_2018':[rent_df.iloc[0:2]['Price_m2_2018'].str.replace('€','').str.replace(',','.').astype(float).mean()]}, columns=['District','Price_m2_2018'])

rent_df2 = rent_df2.append(innenstadt)

rent_df2.sort_values('Price_m2_2018', inplace=True, ascending=False)

rent_df2['District'].replace({'Innenstadt':'Köln-Innenstadt','Rodenkirchen':'Köln-Rodenkirchen','Lindenthal':'Köln-Lindenthal','Ehrenfeld':'Köln-Ehrenfeld','Nippes':'Köln-Nippes','Chorweiler':'Köln-Chorweiler','Porz':'Köln-Porz','Kalk':'Köln-Kalk','Mülheim':'Köln-Mülheim'}, inplace=True)

rent_df2
labels = ['Köln-Lindenthal', 'Köln-Ehrenfeld', 'Köln-Nippes', 'Köln-Innenstadt']



ind = np.arange(len(rent_df2['Price_m2_2018']))  

width = 0.3



fig, ax = plt.subplots(figsize=(16,8))

rects = ax.bar(ind, rent_df2['Price_m2_2018'], width, label=labels, color='#5cb85c')

ax.set_title("Cologne Neighborhoods - Rental Price 2018", fontsize=16)

ax.set_xticks(ind)

ax.set_xticklabels((labels))

plt.ylabel('Rental Price')

plt.xlabel('Districts')

ax.get_yaxis().set_visible(True)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(True)

ax.spines['right'].set_visible(False)



autolabel(rects, "center")



fig.tight_layout()



plt.show()
cgn_df.drop([cgn_df.index[1],cgn_df.index[3]], axis=0, inplace=True)

cgn_df
CLIENT_ID = '2QLU5CN20BM5KPMDHX2QYYBWG13KZAYXAYOIQX0S0300KW01' # your Foursquare ID

CLIENT_SECRET = 'RND2YVJCGI10NSERVO4LCVGFYETQUROC3OCBD0MMK1XPFKKA' # your Foursquare Secret

VERSION = '20200429'

LIMIT = 200

print('Your credentials:')

print('CLIENT_ID: ' + CLIENT_ID)

print('CLIENT_SECRET:' + CLIENT_SECRET)
query_1 = 'Bar'

query_2 = 'Restaurant'

radius = 1250

print('Queries OK!')
url_inne1 = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, cgn_df.iloc[0][4], cgn_df.iloc[0][5], VERSION, query_1, radius, LIMIT)

url_nipp1 = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, cgn_df.iloc[1][4], cgn_df.iloc[1][5], VERSION, query_1, radius, LIMIT)



url_inne2 = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, cgn_df.iloc[0][4], cgn_df.iloc[0][5], VERSION, query_2, radius, LIMIT)

url_nipp2 = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, cgn_df.iloc[1][4], cgn_df.iloc[1][5], VERSION, query_2, radius, LIMIT)





print("URLs created!")
results_inne1 = requests.get(url_inne1).json()

results_nipp1 = requests.get(url_nipp1).json()



results_inne2 = requests.get(url_inne2).json()

results_nipp2 = requests.get(url_nipp2).json()



print("GET Request done, results generated")
venues_inne1 = results_inne1['response']['venues']

venues_nipp1 = results_nipp1['response']['venues']



venues_inne2 = results_inne2['response']['venues']

venues_nipp2 = results_nipp2['response']['venues']



print("JSON File sorted out")
df_inne1 = pd.json_normalize(venues_inne1)

df_inne1['District'] = 'Köln-Innenstadt'

df_inne1['Category'] = 'Bar'

df_inne1.head()
df_nipp1 = json_normalize(venues_nipp1)

df_nipp1['District'] = 'Köln-Nippes'

df_nipp1['Category'] = 'Bar'
df_inne2 = json_normalize(venues_inne2)

df_inne2['District'] = 'Köln-Innenstadt'

df_inne2['Category'] = 'Restaurant'
df_nipp2 = json_normalize(venues_nipp2)

df_nipp2['District'] = 'Köln-Nippes'

df_nipp2['Category'] = 'Restaurant'
df_merged = pd.concat([df_inne1, df_inne2, df_nipp1, df_nipp2])

df_clean = df_merged[['name','Category','District','location.address','location.postalCode','location.lat','location.lng','location.distance']]

df_clean.rename({'name':'Venue','location.address':'Address','location.postalCode':'PostalCode','location.lat':'LocationLatitude','location.lng':'LocationLongitude','location.distance':'LocationDistance'}, axis=1, inplace=True)

df_clean.head()
# create a Stamen Toner map of the world centered around Cologne, Germany

cologne_map = folium.Map(location=[50.936631, 6.958401], zoom_start=10)



# instantiate a feature group for the incidents in the dataframe

locations = folium.map.FeatureGroup()



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, dist in zip(df_clean.LocationLatitude, df_clean.LocationLongitude, df_clean.District):

    if dist == 'Köln-Innenstadt':

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='blue',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6))

    

    elif dist == 'Köln-Nippes':

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='red',

            fill=True,

            fill_color='red',

            fill_opacity=0.6))    



    else:

           pass

            

cologne_map.add_child(locations)
nipp_df = pd.concat([df_nipp1, df_nipp2])

nipp_df = nipp_df[['name','Category','District','location.address','location.postalCode','location.lat','location.lng','location.distance']]

nipp_df.rename({'name':'Venue','location.address':'Address','location.postalCode':'PostalCode','location.lat':'LocationLatitude','location.lng':'LocationLongitude','location.distance':'LocationDistance'}, axis=1, inplace=True)



nipp_df.head()
nipp_df['Category'].value_counts()
inne_df = pd.concat([df_inne1, df_inne2])

inne_df = inne_df[['name','Category','District','location.address','location.postalCode','location.lat','location.lng','location.distance']]

inne_df.rename({'name':'Venue','location.address':'Address','location.postalCode':'PostalCode','location.lat':'LocationLatitude','location.lng':'LocationLongitude','location.distance':'LocationDistance'}, axis=1, inplace=True)



inne_df.head()
inne_df['Category'].value_counts()
neigh_data = {'Bar': [50,17],

        'Restaurant': [50,6]

        }



comp_df = pd.DataFrame(neigh_data, columns = ['Bar','Restaurant'], index=['Köln-Innenstadt','Köln-Nippes'])

comp_df
labels =['Köln-Innenstadt','Köln-Nippes']

bar = comp_df['Bar']

restaurant = comp_df['Restaurant']





ind = np.arange(len(bar))  

width = 0.2



fig, ax = plt.subplots(figsize=(10,8))

rects1 = ax.bar(ind - width, bar, width, label='Bar', color='#5cb85c')

rects2 = ax.bar(ind, restaurant, width, label='Restaurant', color='#5bc0de')





ax.set_title("Neighborhoods Benchmark: Venues Breakdown", fontsize=16)

ax.set_xticks(ind)

plt.ylabel('Number of Venues')

ax.set_xticklabels((labels))

ax.get_yaxis().set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.legend(fontsize=14)



autolabel(rects1, "center")

autolabel(rects2, "center")





fig.tight_layout()



plt.show()
X = df_clean[['LocationLatitude','LocationLongitude']].values[:,1:]

X = np.nan_to_num(X)

cluster_dataset = StandardScaler().fit_transform(X)
num_clusters = 5



k_means = KMeans(init="k-means++", n_clusters=num_clusters, n_init=6)

k_means.fit(cluster_dataset)

labels = k_means.labels_



print(labels)
df_clean["ClusterLabels"] = labels

df_clean.head(5)
# create a Stamen Toner map of the world centered around Cologne, Germany

cologne_map = folium.Map(location=[50.936631, 6.958401], zoom_start=10)



# instantiate a feature group for the incidents in the dataframe

locations = folium.map.FeatureGroup()



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, label in zip(df_clean.LocationLatitude, df_clean.LocationLongitude, df_clean.ClusterLabels):

    if label == 0:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='blue',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6))

    elif label == 1:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='yellow',

            fill_opacity=0.6))

    elif label == 2:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='red',

            fill=True,

            fill_color='red',

            fill_opacity=0.6))

    elif label == 3:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='green',

            fill=True,

            fill_color='green',

            fill_opacity=0.6))

    elif label == 4:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='orange',

            fill=True,

            fill_color='orange',

            fill_opacity=0.6))

    elif label == 5:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='pink',

            fill=True,

            fill_color='pink',

            fill_opacity=0.6))

    elif label == 6:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='purple',

            fill=True,

            fill_color='purple',

            fill_opacity=0.6))

    elif label == 7:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='brown',

            fill=True,

            fill_color='brown',

            fill_opacity=0.6))

    elif label == 8:

        locations.add_child(

        folium.features.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='violet',

            fill=True,

            fill_color='violet',

            fill_opacity=0.6))

        



    else:

           pass

            

cologne_map.add_child(locations)
clust_df = round(df_clean.sort_values('ClusterLabels'),2)

clust_df.head()