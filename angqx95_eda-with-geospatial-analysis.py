import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import MarkerCluster

import json

import urllib



sns.set_style('darkgrid')
PATH = '../input/birdsong-recognition/'



train_df = pd.read_csv(f'{PATH}train.csv')

test_df = pd.read_csv(f'{PATH}test.csv')



print("Train shape: ", train_df.shape, '\t', 'Test shape: ', test_df.shape)
train_df.head()
train_df.info()
train_df['xc_id'].nunique()
print("Columns with missing rows")

print(train_df.isnull().sum().sort_values(ascending=False).head())
print("Total number of unique Bird species: ", train_df['ebird_code'].nunique())

print("Distribution of Bird species in Training set: ")

print(train_df.groupby(['ebird_code']).size().sort_values(ascending=False))
plt.figure(figsize=(18,8))

plt.xticks(rotation=90)

plt.title("Distribution of species across countries")

sns.countplot(data=train_df, x='country')
print("Top 5 countries: \n")

print(train_df.groupby(['country']).size().sort_values(ascending=False).head(5))
world = folium.Map(location=[27.623924, -30.471619], zoom_start=2, min_zoom=2)

mc= MarkerCluster()



for i in range(0,len(train_df)):

    if (train_df.iloc[i]['longitude'] != 'Not specified'):

       mc.add_child(folium.Circle(

          location=[float(train_df.iloc[i]['latitude']), float(train_df.iloc[i]['longitude'])],

          radius=5000,

          color='crimson',

          fill=True,

          fill_color='crimson'

       ))

    

world.add_child(mc)

world
#base map

bird = folium.Map(location=[27.623924, -30.471619], zoom_start=3, min_zoom=2)



killde_df = train_df.loc[train_df['ebird_code'] == 'killde']

greegr_df = train_df.loc[train_df['ebird_code'] == 'greegr']

buffle_df = train_df.loc[train_df['ebird_code'] == 'buffle']

redhea_df = train_df.loc[train_df['ebird_code'] == 'redhea']



#Create feature group

killde = folium.FeatureGroup(name='killde')

greegr = folium.FeatureGroup(name='greegr')

buffle = folium.FeatureGroup(name='buffle')

redhea = folium.FeatureGroup(name='redhea')



def add_point(df, fg, color):

    for i in range(0,len(df)):

        if (df.iloc[i]['longitude'] != 'Not specified'):

           fg.add_child(folium.CircleMarker(

              location=[float(df.iloc[i]['latitude']), float(df.iloc[i]['longitude'])],

              radius=3,

              color=color,

              fill=True,

              fill_color=color

           ))



#Add each species as an overlay 

add_point(killde_df, killde, "red")

add_point(greegr_df, greegr, "green")

add_point(buffle_df, buffle, "blue")

add_point(redhea_df, redhea, "black")



#Add overlay to base map

killde.add_to(bird)

greegr.add_to(bird)

buffle.add_to(bird)

redhea.add_to(bird)



#Add layer control

lc = folium.LayerControl(collapsed=False)

lc.add_to(bird)



bird
# download GEOJson for mapping to choropleth

url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

urllib.request.urlretrieve(url ,'temp.json')
with open('temp.json') as f:

  data = json.load(f)



state_lst = []



for i in data['features']:

    state_lst.append(i['properties']['name'])   #get a list of states from the json
def statein(lst, sent):      #function for state name from 'location' column

    for i in lst:

        if i in sent:

            return i

        

us_df = train_df[train_df['country']== "United States"]

us_df['state'] = us_df['location'].apply(lambda x: statein(state_lst, x)) #extract state from location column into new column



usa_choro = us_df.groupby(['state']).size().reset_index()  #get the numbers of training rows for each USA state

usa_choro.columns = ['state','count']
choro = folium.Map(location=[37.0902, -95.7129], zoom_start=4, min_zoom=2)



folium.Choropleth(

    geo_data='temp.json',

    name='choropleth',

    data=usa_choro,

    bins=9,

    columns=['state','count'], #'state' is the col name required to match with the key from json, 'count' is the value

    key_on='feature.properties.name', #the name property in json will be matched to the 'state'. The names must matched for choropleth to work 

    fill_opacity=0.8,

    line_opacity=0.5,

    fill_color='BuPu',

    legend_name="Distribution of Birds Species in USA").add_to(choro)



folium.LayerControl().add_to(choro)



choro
plt.figure(figsize=(12,8))

plt.title("Distribution of Recordings duration")

sns.distplot(train_df['duration'])
print("Duration of Recordings (in seconds): \n")

print(train_df['duration'].describe())
train_df.loc[train_df['duration']==2283]
with plt.style.context('seaborn-darkgrid'):

    plt.figure(figsize=(18, 10))

    plt.title('Date')

    train_df['date'].value_counts().sort_index().plot()
print("Top 10 dates with most recordings:\n ")

print(train_df['date'].value_counts().sort_values(ascending=False).head(10))
plt.figure(figsize=(12,8))

plt.title("Distribution of ratings")

sns.countplot(data=train_df,x='rating')
plt.figure(figsize=(12,8))

sns.countplot(data=train_df,x='bird_seen')
plt.figure(figsize=(12, 8))

print("Minimum Sampling Rate: ", train_df['sampling_rate'].min(), '\t', "Maximum Sampling Rate: ", train_df['sampling_rate'].max())

train_df['sampling_rate'].value_counts().sort_index().plot()