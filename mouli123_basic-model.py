# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv",nrows=8000000)
data.head()
data.dtypes
data.shape
data_clean=data[(data['dropoff_latitude']<=90) & (data['pickup_latitude']<=90)& (data['pickup_longitude']<=180)& (data['dropoff_longitude']<=180)]
data_clean=data_clean[(data_clean['dropoff_latitude']>-90) & (data_clean['pickup_latitude']>-90)& (data_clean['pickup_longitude']>-180)& (data_clean['dropoff_longitude']>-180)]
data_clean.shape
data_clean.isna().sum()
data_clean.dropna(inplace=True)
import geopy.distance
pickup_long=np.array(data_clean['pickup_longitude'])
pickup_lat=np.array(data_clean['pickup_latitude'])
drop_long=np.array(data_clean['dropoff_longitude'])
drop_lat=np.array(data_clean['dropoff_latitude'])
array=np.zeros(5)
for i in range(5):
    coords_1 = (pickup_lat[i],pickup_long[i])
    coords_2 = (drop_lat[i],drop_long[i])
    array[i]=round(geopy.distance.vincenty(coords_1, coords_2).km,2)
dist=np.zeros(len(data_clean))
for i in range(len(data)):
    coords_1 = (pickup_lat[i],pickup_long[i])
    coords_2 = (drop_lat[i],drop_long[i])
    print(i)
    dist[i]=(geopy.distance.vincenty(coords_1, coords_2).km)
data_clean['distance']=np.round(dist,2)
data_clean.to_csv("uber_dist.csv",index=False)