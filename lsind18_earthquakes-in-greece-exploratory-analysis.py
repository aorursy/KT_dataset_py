import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



eq_df = pd.read_csv('/kaggle/input/earthquakes-in-greece-19012018/EarthQuakes in Greece.csv')

eq_df.head(5)
eq_df.rename(columns={'Date':'Day', 'LATATITUDE (N)':'Lat', 'LONGITUDE  (E)' : 'Long', 'MAGNITUDE (Richter)' : 'Magn' }, inplace=True)

eq_df.info()

eq_df.describe()
eq_df.hist(column='Magn', bins=100)

eq_df[(eq_df['Magn']> 7.3) | (eq_df['Magn']==0.0)].sort_values('Magn', ascending=False)
eq_df = eq_df.loc[eq_df['Magn'] != 0]
import json

from shapely.geometry import mapping, shape

from shapely.prepared import prep

from shapely.geometry import Point



with open('/kaggle/input/greeceborders/Greece_AL2.GeoJson') as json_file:

    data = json.load(json_file)



countries = {}

gr = '' # multipolygon

for feature in data['features']:

    geom = feature['geometry']

    gr = shape(geom)

    countries['Greece'] = prep(gr)
import folium



m = folium.Map([eq_df['Lat'].mean(), eq_df['Long'].mean()], #center of a map

               zoom_start=6, min_zoom = 5, max_zoom = 7) # max zoom is 18; restrict zooms not to scroll much

               

folium.GeoJson(gr).add_to(m) # add gr - multipolygon of greek boundaries

folium.LatLngPopup().add_to(m) # add custom popup of lat/long of selected point



for i in range(0,25): # add markers of first 25 earthquakes of the dataset

    folium.Marker([eq_df.iloc[i]['Lat'], eq_df.iloc[i]['Long']], 

                  popup=eq_df.iloc[i]['Year']).add_to(m) # add popup to markers as an accident year

m
def get_country(row):

    point = Point(row['Long'], row['Lat'])

    for country, geom in countries.items():

        if geom.contains(point):

            return True # country name

    return False # unknown



eq_df['Country'] = eq_df.apply(get_country, axis=1)
eq_gr = eq_df[eq_df.Country == 1]

eq_gr = eq_gr.drop(columns='Country')

eq_gr
eq_minor = eq_gr.loc[(eq_gr['Magn'] > 0) & (eq_gr['Magn'] <=3.9)]

eq_light = eq_gr.loc[(eq_gr['Magn'] > 3.9) & (eq_gr['Magn'] <=4.9)]

eq_moder = eq_gr.loc[(eq_gr['Magn'] > 4.9) & (eq_gr['Magn'] <= 5.9)]

eq_major = eq_gr.loc[(eq_gr['Magn'] > 5.9) & (eq_gr['Magn'] <=7.9)]

eq_great = eq_gr.loc[eq_gr['Magn'] > 7.9]



ax1 = eq_light.plot(kind='scatter', x='Year', y='Month', color='lightgreen', label='Light')

ax2 = eq_moder.plot(kind='scatter', x='Year', y='Month', color='lightblue',  label='Moder', ax=ax1)

ax3 = eq_major.plot(kind='scatter', x='Year', y='Month', color='orange', label='Major', ax=ax1)

ax4 = eq_great.plot(kind='scatter', x='Year', y='Month', color='r', label='Great',  ax=ax1)

ax1.legend(bbox_to_anchor=(1., 1.))
count_y = eq_gr['Year'].value_counts()

count_y.plot(grid=True)
eq_df_20y = eq_gr[eq_gr['Year']>=1998]

pd.crosstab(index=eq_df_20y['Year'], columns='count').plot(kind='bar', figsize=(5,5), grid=True)

eq_df_20y
eq_2014 = eq_gr[eq_gr['Year']==2014]

eq_2014.info()
from folium.plugins import FastMarkerCluster, MarkerCluster

mc = MarkerCluster(name="Marker Cluster")



folium_map = folium.Map([eq_2014['Lat'].mean(), eq_df['Long'].mean()], #center of a map

               zoom_start=6, min_zoom = 5, tiles='Stamen Terrain') # max zoom is 18; restrict zooms not to scroll nuch

               

for index, row in eq_2014[eq_2014.Magn>3.9].iterrows():

    popup_text = "Day: {} <br> Month: {}".format(

                      int(row["Day"]),

                      int(row["Month"])

                      )

    folium.CircleMarker(location=[row["Lat"],row["Long"]],

                        radius= 1.5 * row['Magn'],

                        color="red",

                        popup=popup_text,

                        fill=True).add_to(mc)



mc.add_to(folium_map)



folium.LayerControl().add_to(folium_map)

folium_map
eq_light14 = eq_light[eq_light.Year==2014]

eq_moder14 = eq_moder[eq_moder.Year==2014]

eq_major14 = eq_major[eq_major.Year==2014]
m = folium.Map([eq_2014['Lat'].mean(), eq_df['Long'].mean()], #center of a map

               zoom_start=6, min_zoom = 5) # max zoom is 18;

              

for i in range(0,len(eq_light14)): # add markers of eq with diff magn 

    folium.Circle(

        radius=1000 * eq_light14.iloc[i]['Magn'],

        location=[eq_light14.iloc[i]['Lat'], eq_light14.iloc[i]['Long']],

        popup="D: {} <br> Mo: {}".format(

                      int(row["Day"]),

                      int(row["Month"])),

        color='green',

        fill=False,

    ).add_to(m) 



for i in range(0,len(eq_moder14)): # add markers of eq with diff magn 

    folium.Circle(

        radius=1000 * eq_moder14.iloc[i]['Magn'],

        location=[eq_moder14.iloc[i]['Lat'], eq_moder14.iloc[i]['Long']],

        popup="D: {} <br> Mo: {}".format(

                      int(row["Day"]),

                      int(row["Month"])),

        color='blue',

        fill=False,

    ).add_to(m) 



for i in range(0,len(eq_major14)): # add markers of eq with diff magn 

    folium.Circle(

        radius=1000 * eq_major14.iloc[i]['Magn'],

        location=[eq_major14.iloc[i]['Lat'], eq_major14.iloc[i]['Long']],

        popup="D: {} <br> Mo: {}".format(

                      int(row["Day"]),

                      int(row["Month"])),

        color='red',

        fill=False,

    ).add_to(m)

m