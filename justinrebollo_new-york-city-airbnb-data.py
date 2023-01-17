import numpy as np 

import pandas as pd

import plotly.express as px

import folium

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Create the dataframe

bnb_data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', dtype=str)



#understand data

bnb_data.describe()
#Get column names 

bnb_data.columns



#normalize price for plotting

bnb_data['price'] = bnb_data['price'].astype(int) 

bnb_data['price'] = bnb_data['price']/250  #radius too large without normalizing 

#Fix neighborhoods and columns for proper plotting

bnb_data.rename(columns = {'neighbourhood_group':'boroname'}, inplace = True)

bnb_data.drop(['host_name',

       'neighbourhood', 'room_type',

       'minimum_nights', 'number_of_reviews', 'last_review',

       'reviews_per_month', 'calculated_host_listings_count',

       'availability_365'], axis=1)
#Visualize Price Data 



import folium



center_lat= 40.7128

center_long= -74.0060



bnb_map = folium.Map(location=[center_lat, center_long], zoom_start=13)





for i, r in bnb_data.iterrows():

    folium.CircleMarker(

        location= [r['latitude'], r['longitude']],

        radius= [r['price']],

        color='red',

        fill = True,

        fill_color = 'blue',

        fill_opacity = 0.6

    ).add_to(bnb_map)



bnb_map




