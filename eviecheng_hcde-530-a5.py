# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

baywheels = pd.read_csv("../input/202001-baywheels-tripdata.csv")

baywheels.head()
baywheels['user_type'].value_counts(normalize=True)
baywheels['user_type'].value_counts().plot(title='How many frequent members versus casual customers', kind='pie')
baywheels['duration_sec'].describe()
baywheels.plot(y='duration_sec', title='How long do customers bike (in seconds)')
import mpu

for i in range(len(baywheels)):

    baywheels.at[i, 'distance'] = mpu.haversine_distance((baywheels.at[i, 'start_station_latitude'], baywheels.at[i, 'start_station_longitude']), (baywheels.at[i, 'end_station_latitude'], baywheels.at[i, 'end_station_longitude']))

baywheels['distance'].describe()
baywheels.plot(y='distance', title='How far do customers bike (in kilometers)')
baywheels['start_station_name'].value_counts()[:20].plot(kind="bar", title="Top 20 stations where customers start their trips")
baywheels['end_station_name'].value_counts()[:20].plot(kind="bar", title="Top 20 stations where customers end their trips")
n = 20

popularstartstations = baywheels['start_station_name'].value_counts()[:n].index.tolist()

popularendstations = baywheels['end_station_name'].value_counts()[:n].index.tolist()



def intersection(lst1, lst2): 

    lst3 = [value for value in lst1 if value in lst2] 

    return lst3 

  

print(len(intersection(popularstartstations, popularendstations)))

print(intersection(popularstartstations, popularendstations))