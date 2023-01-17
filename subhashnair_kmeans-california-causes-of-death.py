import types

import pandas as pd



dfD = pd.read_csv('../input/deaths-by-zip-code-by-cause-of-death-1999-current.csv')

dfD.head()

#California Leading Causes of Death by ZIP Code - https://healthdata.gov/dataset/leading-causes-death-zip-code
#Change column names

dfD = dfD.rename({'ZIP Code':'ZipCode', 'Causes of Death':'CausesofDeath'}, axis=1)

dfD.head()
dfD['ZipCode'] = dfD['ZipCode'].apply(str)

dfD.dtypes
dfD = dfD[~(dfD.ZipCode.str.len() < 5)]
dfD.groupby(['ZipCode','CausesofDeath']).Count.sum().head()
df_data_0 = pd.read_csv('../input/us-zip-code-latitude-and-longitude.csv')

df = df_data_0.drop(['City','State','Timezone','Daylight savings time flag','geopoint'], axis = 1) 

df.head()



df = df.rename({'Zip':'ZipCode'}, axis=1)

df['ZipCode'] = df['ZipCode'].apply(str)

df.head()
dfD = pd.merge(df, dfD, how='outer', on=['ZipCode'])

dfD.reset_index(drop=True, inplace=True)

dfD.head()
import numpy as np

dfD = dfD[np.isfinite(dfD['Year'])]

dfD = dfD[np.isfinite(dfD['Latitude'])]

dfD.shape
dfD['Year'] = dfD['Year'].apply(int)

dfD['Count'] = dfD['Count'].apply(int)

dfD['CausesofDeath'] = dfD['CausesofDeath'].apply(str)

dfD = dfD[dfD['Count'] > 0]

dfD.loc[dfD['ZipCode'] == "92127"].head()

dfD1 = dfD.loc[dfD['ZipCode'] == "90210"]

dfD1.groupby(['ZipCode','CausesofDeath'])['Count'].sum().nlargest(10)
dfD1 = dfD.loc[dfD['CausesofDeath'] == "SUI"]

dfD1.groupby(['ZipCode','CausesofDeath'])['Count'].sum().nlargest(10)

import numpy as np # library to handle data in a vectorized manner



import pandas as pd # library for data analsysis

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



import json # library to handle JSON files



!conda install -c conda-forge geopy --yes 

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values



import requests # library to handle requests

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe



# Matplotlib and associated plotting modules

import matplotlib.cm as cm

import matplotlib.colors as colors



# import k-means from clustering stage

from sklearn.cluster import KMeans



!conda install -c conda-forge folium=0.5.0 --yes 

import folium # map rendering library



print('Libraries imported.')
address = 'California, USA'



geolocator = Nominatim(user_agent="on_explorer")

location = geolocator.geocode(address)

latitude = location.latitude

longitude = location.longitude

print('The geographical coordinates of California are {}, {}.'.format(latitude, longitude))
dfD.reset_index(drop=True, inplace=True)

dfD.head()
dfD1 = dfD.loc[(dfD.Year == 2016) & (dfD.CausesofDeath == "NEP")]

dfD1.shape

# create map of California with causes of death as "NEP" using latitude and longitude values

map_dfD = folium.Map(location=[latitude, longitude], zoom_start=6)



# add markers to map

for lat, lng, label in zip(dfD1['Latitude'], dfD1['Longitude'], dfD1['CausesofDeath']):

    label = folium.Popup(label, parse_html=True)

    folium.CircleMarker(

        [lat, lng],

        radius=5,

        popup=label,

        color='blue',

        fill=True,

        fill_color='#3186cc',

        fill_opacity=0.7,

        parse_html=False).add_to(map_dfD)  

    

map_dfD
dfD1 = dfD.groupby(['ZipCode','CausesofDeath']).Count.sum().reset_index()

dfD2 = dfD1.pivot(index='ZipCode', columns='CausesofDeath').reset_index()

dfD2.fillna(0, inplace=True)

dfD3 = pd.DataFrame(dfD2.to_records())

dfD3.columns = ['index','ZipCode', 'ALZ', 'CAN', 'CLD', 'DIA' , 'HOM' , 'HTD' , 'HYP' , 'INJ' , 'LIV' , 'NEP' , 'OTH' , 'PNF' , 'STK' , 'SUI']

dfD3.drop('index', axis=1, inplace=True)

dfD3.head()
def return_most_common_deaths(row, num_top_deaths):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_deaths]
num_top_deaths = 14

indicators = ['st', 'nd', 'rd']



# create columns according to number of top deaths

columns = ['ZipCode']

for ind in np.arange(num_top_deaths):

    try:

        columns.append('{}{} Most Common cause of Death'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common cause of Death'.format(ind+1))



# create a new dataframe

Causesofdeath_sorted = pd.DataFrame(columns=columns)

Causesofdeath_sorted['ZipCode'] = dfD3['ZipCode']



for ind in np.arange(dfD3.shape[0]):

   Causesofdeath_sorted.iloc[ind, 1:] = return_most_common_deaths(dfD3.iloc[ind, :], num_top_deaths)

Causesofdeath_sorted.sort_values(Causesofdeath_sorted.columns[0]).head()
# set number of clusters

num_top_deaths = 5

kclusters = 5



dfD3_clustering = dfD3.drop('ZipCode', 1)

#toronto_grouped_clustering

# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dfD3_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:5] 

# add clustering labels

Causesofdeath_sorted.insert(0, 'Cluster Labels', kmeans.labels_)



# merge dfD3 with df to add latitude/longitude for each neighborhood

df = df.join(Causesofdeath_sorted.set_index('ZipCode'), on='ZipCode')

df = df.dropna()

df = df.reset_index(drop=True)
df['Cluster Labels'] = df['Cluster Labels'].astype(int)

df.head()
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=5)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



countmap = 0

# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(df['Latitude'], df['Longitude'], df['ZipCode'], df['Cluster Labels']):

    

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster - 1],

        fill=True,

        fill_color=rainbow[cluster - 1],

        fill_opacity=0.7).add_to(map_clusters)

    countmap = countmap + 1

    #Folium crashes on the complete dataset (likely because I have a 'free' Watson account). So, I am limiting to 50 for demonstration purposes!!!!

    if countmap == 50:

            break

map_clusters
df.loc[df['Cluster Labels'] == 0, df.columns[[0] + list(range(4, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 1, df.columns[[0] + list(range(4, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 2, df.columns[[0] + list(range(4, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 3, df.columns[[0] + list(range(4, df.shape[1]))]]
df.loc[df['Cluster Labels'] == 4, df.columns[[0] + list(range(4, df.shape[1]))]]