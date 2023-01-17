import pandas as pd  #pandas for using dataframe and reading csv 

import numpy as np   #numpy for vector operations and basic maths 

import urllib        #for url stuff

import re            #for processing regular expressions

import datetime      #for datetime operations

import calendar      #for calendar for datetime operations

import time          #to get the system time

import scipy         #for other dependancies

from sklearn.cluster import KMeans # for doing K-means clustering

from haversine import haversine # for calculating haversine distance

import math          #for basic maths operations

import seaborn as sns #for making plots

import matplotlib.pyplot as plt # for plotting

import os                # for os commands

import nltk

from nltk.corpus import stopwords

import string

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes
air_reserve = pd.read_csv("../input/air_reserve.csv")

air_store = pd.read_csv("../input/air_store_info.csv")

air = air_reserve.merge(air_store, on = 'air_store_id', how = 'left')

air_visit = pd.read_csv("../input/air_visit_data.csv")

air = air.merge(air_visit, on = 'air_store_id', how = 'left')

date_info = pd.read_csv("../input/date_info.csv")

air = air.merge(date_info, left_on='visit_date', right_on = 'calendar_date', how = 'left')

# air.head()
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")

store_id_rel = pd.read_csv("../input/store_id_relation.csv")

hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")

hpg = hpg_reserve.merge(hpg_store_info, on = 'hpg_store_id', how = 'left')

hpg = hpg.merge(store_id_rel, on = 'hpg_store_id', how = 'left')

#train = hpg.merge(air, on = 'air_store_id', how = 'left')

#train.head()

hpg.head()
air.head()

genre_summary = pd.DataFrame(air.groupby('air_genre_name')['air_genre_name'].count())

genre_summary.reset_index(drop = True)

genre_summary = genre_summary.sort_values('air_genre_name', ascending = False)
import folium

def show_fmaps(train_data, path=1):

    """function to generate map and add the pick up and drop coordinates

    1. Path = 1 : Join pickup (blue) and drop(red) using a straight line

    """

    full_data = train_data

    summary_full_data = pd.DataFrame(full_data.groupby('air_genre_name')['air_store_id'].count())

    summary_full_data.reset_index(inplace = True)

    summary_full_data = summary_full_data.loc[summary_full_data['air_store_id']>70000]

    map_1 = folium.Map(location=[35.767937, 139.982155], zoom_start=10,tiles='OpenStreetMap') # manually added centre

    new_df = train_data.loc[train_data['air_genre_name'].isin(summary_full_data.air_genre_name.tolist())].sample(50)

    new_df.reset_index(inplace = True, drop = True)

    for i in range(new_df.shape[0]):

        pick_long = new_df.loc[new_df.index ==i]['longitude'].values[0]

        pick_lat = new_df.loc[new_df.index ==i]['latitude'].values[0]

        #dest_long = new_df.loc[new_df.index ==i]['dropoff_longitude'].values[0]

        #dest_lat = new_df.loc[new_df.index ==i]['dropoff_latitude'].values[0]

        folium.Marker([pick_lat, pick_long]).add_to(map_1)

        #folium.Marker([dest_lat, dest_long]).add_to(map_1)

    return map_1
osm = show_fmaps(air, path=1)

osm