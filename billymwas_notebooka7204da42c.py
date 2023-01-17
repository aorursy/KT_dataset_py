# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
#import dataset
df = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/train.zip')

#inspect dataset
df.head()
#inspect the types

df.dtypes
#inspect last 5 rows
df.tail()
#check for any missing value
df.isna().any()
df.describe()
df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime']=pd.to_datetime(df['dropoff_datetime'])
from haversine import haversine
df['trip_distance'] = df.apply(lambda x: haversine((x.pickup_latitude,x.pickup_longitude),(x.dropoff_latitude,x.dropoff_longitude)),axis=1)
df.head()
#total number of trips
df['id'].count()
#unique vendors
df['vendor_id'].unique().shape[0]
t = df.groupby('passenger_count')[['passenger_count']].count()
t.plot(kind='bar')
# - Early morning (4 hrs to 8 hrs)
# - Morning (8 hrs to 12 hrs) , 
# - Afternoon (12 hrs to 17 hrs) , 
# - Evening (17 hrs to  20 hrs),
# - Night (20 hrs to 0 hrs)
# - Mid night (0 hrs to 4hrs)


def period(a):
    if a in range(0,4):
        return 'Mid night'
    elif a in range(4,8):
        return "Early morning"
    elif a in range(8,12):
        return "Morning"
    elif a in range(12,17):
        return "Afternoon"
    elif a in range(17,20):
        return "Evening"
    else:
        return "Night"

df['dropoff_timezone'] = df.apply(lambda x: period(x['dropoff_datetime'].hour),axis=1)

df.head()
df.groupby('dropoff_timezone')['dropoff_timezone'].count().plot(kind='bar')
clean_df = df[df['trip_distance'] > 0.0]
clean_df = df[~df['trip_duration'].isin(df.nlargest(4,'trip_duration')['trip_duration'])]
clean_df.nlargest(50,'trip_distance')
clean_df.set_index('pickup_datetime')
import matplotlib.pyplot as plt
fig,ax = plt.subplots()

# ax.scatter(clean_df['trip_duration'],clean_df['trip_distance'],color='blue')
# ax.set_xlabel('trip distance')
# ax.set_ylabel('trip duration')
# ax.boxplot([clean_df['trip_duration']])

ax.scatter(clean_df.index,clean_df[['trip_duration']],color='blue',label='trip duration')
ax.set_ylabel('trip duration',color='blue')
ax.set_xlabel('time')
ax.tick_params('y', colors='blue')
ax.legend()
ax2 = ax.twinx()
ax2.scatter(clean_df.index,clean_df['trip_distance'],color='red',label='trip distance')
ax2.set_ylabel('trip distance',color='red')
ax2.tick_params('y',colors='red')
fig.set_size_inches(7,7)
plt.show()
trip_per_day = df.groupby(df['pickup_datetime'].dt.date)['id'].count()
trip_per_day.plot(ylabel='total trips per day')
trip_duration_per_day = df.groupby(df['pickup_datetime'].dt.date)['trip_distance'].agg(['sum'])
trip_duration_per_day.plot(ylabel='trip duration per day')
total_passengers_per_day = df.groupby(df['pickup_datetime'].dt.date)['passenger_count'].agg(['sum'])
total_passengers_per_day.plot(ylabel='total passengers per day')
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

geometry = [Point(xy) for xy in zip(df['pickup_longitude'], df['pickup_latitude'])]
gdf = GeoDataFrame(df, geometry=geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We restrict to North America.
ax = world[world.continent == 'North America'].plot(
    color='white', edgecolor='black',figsize=(8, 8))

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color='red')

plt.show()