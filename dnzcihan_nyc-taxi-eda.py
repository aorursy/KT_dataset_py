import pandas as pd 

from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt

import seaborn as sns 



data = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

data.head()

print(data.describe(),data.info())

data['trip_duration'] = data['trip_duration']/60

data['trip_duration'] = round(data.trip_duration,2)

data.head()
from math import radians, sin, cos, sqrt, asin

def haversine(columns):

  lat1, lon1, lat2, lon2 = columns

  R = 6372.8 # Earth radius in kilometers

 

  dLat = radians(lat2 - lat1)

  dLon = radians(lon2 - lon1)

  lat1 = radians(lat1)

  lat2 = radians(lat2)

 

  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2

  c = 2*asin(sqrt(a))

 

  return R * c

cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']

distances = data[cols].apply(

    lambda x: haversine(x),axis = 1

)

data['distance'] = distances.copy()

data['distance'] = round(data.distance,2)

data.head()
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

data['pu_hour'] = data['pickup_datetime'].dt.hour

data['pu_day'] = data['pickup_datetime'].dt.dayofyear

data['pu_wday'] = data['pickup_datetime'].dt.dayofweek

data['pu_month'] = data['pickup_datetime'].dt.month

data.head()

rush_hour_morning = data[(data['pu_hour'] >= 7) & (data['pu_hour'] < 9)]

afternoon = data[(data['pu_hour'] >= 9) & (data['pu_hour'] < 16)]

rush_hour_evening = data[(data['pu_hour'] >= 16) & (data['pu_hour'] < 18)]

evening = data[(data['pu_hour'] >= 18) & (data['pu_hour'] <= 23)]

latenight =data[(data['pu_hour'] >=0 ) & (data['pu_hour'] < 7)]



print("7am and 9 am average distance :",rush_hour_morning.distance.mean(),

"9am and 4 P.M average distance :",afternoon.distance.mean(),

"4 PM and 6 PM average distance : ",rush_hour_evening.distance.mean(),

"6 pm and 11pm average distance :",evening.distance.mean(),

"11pm and 7 am average distance :",latenight.distance.mean(),)



passengers = data['passenger_count'].value_counts().sort_index()

passengers.plot(kind = 'bar',logy = True)

plt.xlabel('Number of passengers')

plt.ylabel('Frequency')

plt.title('Distribution of passenger counts, log scaling')

plt.show()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

ax.scatter(data['pickup_longitude'],data['pickup_latitude'], s=0.02, alpha=1)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

ax.scatter(data['dropoff_longitude'],data['dropoff_latitude'], s=0.02, alpha=1)
byWday = rush_hour_morning.groupby('pu_wday').count()['trip_duration']

byWday.plot()

plt.tight_layout()

byWday = afternoon.groupby('pu_wday').count()['trip_duration']

byWday.plot()

plt.tight_layout()
byWday = rush_hour_evening.groupby('pu_wday').count()['trip_duration']

byWday.plot()

plt.tight_layout()
byWday = evening.groupby('pu_wday').count()['trip_duration']

byWday.plot()

plt.tight_layout()
byWday = latenight.groupby('pu_wday').count()['trip_duration']

byWday.plot()

plt.tight_layout()
byWday = latenight.groupby('pu_wday').count()['distance']

byWday.plot()

plt.tight_layout()
sns.countplot(x='pu_wday',data=data,hue='vendor_id')

plt.tight_layout()
data.corr()
a=data.groupby('passenger_count').count()['distance']

a.plot()