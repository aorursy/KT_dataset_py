import requests

import os

import pandas as pd

import seaborn as sns

import numpy as np

import folium

from random import randint

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

%matplotlib inline
BOURNEMOUTH = (50.721680, -1.878530)

ZOOM = 15



COL_LAT = 'Latitude'

COL_LNG = 'Longitude'

COL_VENUE_NAME = 'Venue Name'

COL_VENUE_LAT = 'Venue Latitude'

COL_VENUE_LNG = 'Venue Longitude'

COL_VENUE_CAT = 'Venue Category'

COL_VENUE_GRP = 'Venue Group'

COL_VENUE_CLS = 'Venue Cluster'



DRINK = 'Drink'

ENTERTAINMENT = 'Entertainment'

FOOD = 'Food'

HOTEL = 'Hotel'

SHOPPING = 'Shopping'

TRANSPORT = 'Transport'
path_data = os.path.join('..', 'input', 'bournemouth_venues.csv')



df = pd.read_csv(path_data)



print(df.shape)

df.head()
def generate_map(df, lat, lng, zoom, col_lat, col_lng,

                 col_popup=None, popup_colors=False, def_color='red',

                 tiles='cartodbpositron'):

    folmap = folium.Map(location=[lat, lng], zoom_start=zoom, tiles=tiles)

    

    popup = list(df[col_popup].unique())

    

    if popup_colors:

        colors = make_color_palette(len(popup))

    

    for index, row in df.iterrows():

        folium.CircleMarker(

            location=(row[col_lat], row[col_lng]),

            radius=6,

            popup=row[col_popup] if col_popup is not None else '',

            fill=True,

            color=colors[popup.index(row[col_popup])] if popup_colors else def_color,

            fill_opacity=0.6

            ).add_to(folmap)

    

    return folmap
def make_color_palette(size, n_min=50, n_max=205):

    r = lambda: hex(randint(0, 255))[2:]

    colors = []

    

    while len(colors) < size:

        c = '#{}{}{}'.format(r(), r(), r())

        

        if c not in colors:

            colors.append(c)

    

    return colors
generate_map(df, BOURNEMOUTH[0], BOURNEMOUTH[1], ZOOM, COL_VENUE_LAT, COL_VENUE_LNG, col_popup=COL_VENUE_NAME)
venue_cat = df[COL_VENUE_CAT].unique()

venue_cat.sort()



print('Venue count:', len(venue_cat))

venue_cat
def change_group(df, grp_from_list, grp_to):

    for grp_from in grp_from_list:

        df.loc[df[COL_VENUE_GRP] == grp_from, COL_VENUE_GRP] = grp_to
# Quickly set venue groups to the last word in each venue category

df[COL_VENUE_GRP] = df[COL_VENUE_CAT].str.split(' ').str[-1]



# Remove the train station platform venue because we already have the nearby train station as a venue

df = df[df[COL_VENUE_GRP] != 'Platform']



# Change the crude, last-word groups into more high-level groups

change_group(df,

             ['Bar', 'Brewery', 'Nightclub', 'Pub'],

             DRINK)



change_group(df,

             ['Aquarium', 'Beach', 'Center', 'Garden', 'Gym', 'Lookout',

              'Multiplex', 'Museum', 'Outdoors', 'Park', 'Theater'],

             ENTERTAINMENT)



change_group(df,

             ['Café', 'Diner', 'House', 'Joint', 'Place', 'Restaurant'],

             FOOD)



change_group(df,

             ['Plaza', 'Shop', 'Store'],

             SHOPPING)



change_group(df,

             ['Station', 'Stop'],

             TRANSPORT)



venue_grp = df[COL_VENUE_GRP].unique()

venue_grp.sort()



print('Group count:', len(venue_grp))

venue_grp
generate_map(df, BOURNEMOUTH[0], BOURNEMOUTH[1], ZOOM, COL_VENUE_LAT, COL_VENUE_LNG, col_popup=COL_VENUE_GRP, popup_colors=True)
df_latlng = df[[COL_VENUE_LAT, COL_VENUE_LNG]]

df_latlng.head()
latlng = StandardScaler().fit_transform(np.nan_to_num(df_latlng))

latlng[:5]
dbscan = DBSCAN(eps=0.2, min_samples=3)

dbscan.fit(latlng)



print('labels:', np.unique(dbscan.labels_))
df[COL_VENUE_CLS] = dbscan.labels_

df.head()
generate_map(df, BOURNEMOUTH[0], BOURNEMOUTH[1], ZOOM, COL_VENUE_LAT, COL_VENUE_LNG, col_popup=COL_VENUE_CLS, popup_colors=True)
df[COL_VENUE_CLS].plot(kind='hist')
df_0 = df.loc[df[COL_VENUE_CLS] == 0]

df_2 = df.loc[df[COL_VENUE_CLS] == 2]



print('df_0:', df_0.shape)

print('df_2:', df_2.shape)
df_0[COL_VENUE_GRP].value_counts().plot(kind='barh', title='df_0: {} venues'.format(df_0.shape[0]))
df_2[COL_VENUE_GRP].value_counts().plot(kind='barh', title='df_2: {} venues'.format(df_2.shape[0]))