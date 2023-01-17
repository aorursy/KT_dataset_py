import os

import pandas as pd

import numpy as np

from math import *
sbuxlocs = pd.read_csv('../input/directory.csv', encoding = "ISO-8859-1")

sbuxlocs.columns = [c.replace(' ', '_') for c in sbuxlocs.columns]
def haversine_dist(df):

    lon1 = df['Longitude_x']

    lat1 = df['Latitude_x']

    lon2 = df['Longitude_y']

    lat2 = df['Latitude_y']

    lon1, lat1, lon2, lat2 = np.deg2rad([lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a)) 

    return 3956 * c #replace 3956 with 6367 for km
s = sbuxlocs.loc[:,['Store_Number','Longitude','Latitude']]

s['key'] = 1
numChunks = 100

minDist = 100000.0



# Chunk process into smaller sizes

for i in range(0, numChunks):

    startRow = i * len(s.index)/numChunks

    endRow = (i+1) * len(s.index)/numChunks - 1

    progress = round((int(endRow) / (len(s.index)-1)) * 100, 2)

    print (str(progress) + '%')

    

    # Partition locations list into chunks

    s1 = s.loc[int(startRow):len(s.index),:]

    s2 = s.loc[int(startRow):int(endRow),:]

    

    # Generate cartesian product

    s3 = pd.merge(s1, s2, on='key')

    del s3['key']

    

    # Calculate cartesian distance

    distVals = haversine_dist(s3).tolist()

    s3['distance'] = pd.Series(distVals).values

    

    # Store closest pair so far

    if s3.loc[s3.distance>0,:].distance.min() < minDist:

        minDist = s3.loc[s3.distance>0,:].distance.min()

        smin = s3.loc[s3.distance>0,:].distance.argmin()

        locationXStoreNum = s3.loc[smin,'Store_Number_x']

        locationXAddress = sbuxlocs.loc[sbuxlocs.Store_Number == s3.loc[smin,'Store_Number_x'],'Street_Address'].to_string(index=False)

        locationXCity = sbuxlocs.loc[sbuxlocs.Store_Number == s3.loc[smin,'Store_Number_x'],'City'].to_string(index=False)

        locationYStoreNum = s3.loc[smin,'Store_Number_y']

        locationYAddress = sbuxlocs.loc[sbuxlocs.Store_Number == s3.loc[smin,'Store_Number_y'],'Street_Address'].to_string(index=False)

        locationYCity = sbuxlocs.loc[sbuxlocs.Store_Number == s3.loc[smin,'Store_Number_y'],'City'].to_string(index=False)

        print('The shortest distance so far is ' +  str(minDist) + ' miles')

    

    del s1

    del s2

    del s3
print('The shortest distance is ' + str(minDist) + ' miles between Store number ' 

      + locationXStoreNum + ' at ' + locationXAddress + ', ' + locationXCity 

      + ' and ' 

      + locationYStoreNum + ' at ' + locationYAddress + ', ' + locationYCity 

     )

    