!pip install reverse_geocoder
!pip install --upgrade pip

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import geopy.distance as geo

import reverse_geocoder as rg

import datetime as dt

import time

import calendar

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/2014_15_taxi.csv")

def get_pickup_loc(df1):

    coordinates = list()

    max_count = df1.pickup_latitude.count()

    for x in range (0,max_count):

        temp =(df1['pickup_latitude'][x],df1['pickup_longitude'][x])

        coordinates.append(temp)



    #address = list

    address = pd.DataFrame()

    add_loc = list()

    add_country = list()

    

    result = rg.search(coordinates) 

    for x in range(0,max_count):

        add_loc.append(result[x]['name'])

        add_country.append(result[x]['cc'])

        

    address['loc'] = add_loc

    address['Country'] = add_country

    #.append(result[x]['name'], result[x]['cc'])

    return address



      
def get_dropoff_loc(df1):

    coordinates = list()

    max_count = df1.dropoff_latitude.count()

    for x in range (0,max_count):

        temp =(df1['dropoff_latitude'][x],df1['dropoff_longitude'][x])

        coordinates.append(temp)



    address = pd.DataFrame()

    add_loc = list()

    add_country = list()

    result = rg.search(coordinates) 

      

    for x in range(0,max_count):

        add_loc.append(result[x]['name'])

        add_country.append(result[x]['cc'])

        

    address['loc'] = add_loc

    address['Country'] = add_country

        #address.append(result[x]['name'])

    return address

unwanted_indices = df[ (abs(df['pickup_latitude']) > 90) | 

                       (abs(df['dropoff_latitude']) > 90) 

                     ].index





df.drop(list(unwanted_indices), inplace=True)

unwanted_indices = df[ (abs(df['pickup_longitude']) > 180) | 

                       (abs(df['dropoff_longitude']) > 180) 

                     ].index



df.drop(list(unwanted_indices), inplace=True)



df.shape
unwanted_indices = df[ (abs(df['pickup_latitude']) == 0) | 

                       (abs(df['dropoff_latitude']) == 0) |

                       (abs(df['pickup_longitude']) == 0) | 

                       (abs(df['dropoff_longitude']) == 0)].index

df.drop(list(unwanted_indices), inplace=True)

df = df.reset_index(drop=True)









unwanted_indices = df[df['dropoff_longitude'].isna()].index

df.drop(list(unwanted_indices), inplace=True)

df = df.reset_index(drop=True)



df.shape
df['distance'] = list(  map( lambda x1,x2,x3,x4: 

                               geo.distance( (x3,x1), (x4,x2) ).miles,

                               df['pickup_longitude'], df['dropoff_longitude'],

                               df['pickup_latitude'],  df['dropoff_latitude'] ) )
if __name__=="__main__":

    coordinates = list()

    location_det = pd.DataFrame()

    location_det = get_pickup_loc(df)

    df['pickup_location'] = location_det['loc']

    df['pickup_country'] = location_det['Country']

    #get drop off details

    location_det = pd.DataFrame()

    location_det = get_dropoff_loc(df)

    df['dropoff_location'] = location_det['loc']

    df['dropoff_country'] = location_det['Country']
unwanted_indices = df[(df['pickup_country'] != 'US') | (df['dropoff_country'] != 'US')].index

df.shape

df.to_csv('2014_15_txdet3.csv',index=None)