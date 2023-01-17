import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import time

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
!pip install googlemaps
import googlemaps
gmaps = googlemaps.Client(key='AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY')
for i, j in train_df.iterrows():

    print(j.lat)

    print(j.long)

    g = gmaps.reverse_geocode((j.lat, j.long))

    print(g[0]['formatted_address'])

    break
train_df['address'] = None

test_df['address'] = None
combine = [train_df, test_df]
start = time.time()

for dataset in combine:

    count = 0 

    for i, j in dataset.iterrows():

        g = gmaps.reverse_geocode((j.lat, j.long))

        dataset['address'][i] = g[0]['formatted_address']

        count +=1

        if count % 1000 == 0:

            print(count)

        

print("Finish convert", time.time()-start)
train_df.address.head(10)
train_df.to_csv('new_train.csv')

test_df.to_csv('new_test.csv')