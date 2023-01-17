# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', nrows=2_000_000, parse_dates=['pickup_datetime'] )
# to read the first three rows of training dataset

df_train.head(3)
#check the datatypes

df_train.dtypes
#check the statictics of the features

df_train.describe()
#check for missing values in train data

df_train.isnull().sum().sort_values(ascending=False)
#drop the missing values

df_train = df_train.drop(df_train[df_train.isnull().any(1)].index, axis = 0)
df_train.describe()
df_train.boxplot(column='fare_amount')
df_train.boxplot(column='passenger_count')
from collections import Counter

Counter(df_train['fare_amount']<0)
df_train=df_train.drop(df_train[df_train['fare_amount']<0].index, axis=0)

df_train.shape
df_train=df_train.drop(df_train[df_train['passenger_count']>6].index, axis=0)

df_train['passenger_count'].describe()
#checking the pickup_latitude 

df_train['pickup_latitude'].describe()
len(df_train[df_train['pickup_latitude']<-90])
len(df_train[df_train['pickup_latitude']>90])
df_train['pickup_latitude'].shape
df_train=df_train.drop(((df_train[df_train['pickup_latitude']<-90])|(df_train[df_train['pickup_latitude']>90])).index, axis=0)
df_train['pickup_latitude'].shape
df_train['pickup_longitude'].shape
df_train=df_train.drop(((df_train[df_train['pickup_longitude']<-180])|(df_train[df_train['pickup_longitude']>180])).index, axis=0)
df_train['pickup_longitude'].shape
df_train.describe()
df_train=df_train.drop(((df_train[df_train['dropoff_latitude']<-90])|(df_train[df_train['dropoff_latitude']>90])).index, axis=0)
df_train=df_train.drop(((df_train[df_train['dropoff_longitude']<-180])|(df_train[df_train['dropoff_longitude']>180])).index, axis=0)
df_train.describe()
df_train['key']=pd.to_datetime(df_train['key'])

df_train['pickup_datetime']=pd.to_datetime(df_train['pickup_datetime'])
df_train.dtypes
data = [df_train]

for i in data:

    i['year']=i['pickup_datetime'].dt.year

    i['month']=i['pickup_datetime'].dt.month

    i['date']=i['pickup_datetime'].dt.date

    i['day of week']=i['pickup_datetime'].dt.dayofweek

    i['hour']=i['pickup_datetime'].dt.hour
df_train.head(3)
plt.figure(figsize=(15,7))

plt.hist(df_train['passenger_count'], bins=15)

plt.xlabel('Number of Passengers')

plt.title('Fare rates based on the number of passengers')

plt.show()
plt.figure(figsize=(15,7))

plt.scatter(x=df_train['passenger_count'], y=df_train['fare_amount'], s=1.5)

plt.xlabel('Number of Passengers')

plt.ylabel('Fare amount')

plt.title('Fare rates based on the number of passengers')

plt.show()
plt.figure(figsize=(15,7))

plt.scatter(x=df_train['year'], y=df_train['fare_amount'], s=1.5)

plt.xlabel('Year')

plt.ylabel('Fare amount')

plt.title('Year wise taxi fares details')

plt.show()
plt.figure(figsize=(15,7))

plt.scatter(x=df_train['month'], y=df_train['fare_amount'], s=1.5)

plt.xlabel('Month')

plt.ylabel('Fare amount')

plt.title('Month wise taxi fares details')

plt.show()
plt.figure(figsize=(15,7))

plt.scatter(x=df_train['day of week'], y=df_train['fare_amount'], s=1.5)

plt.xlabel('Day of week')

plt.ylabel('Fare amount')

plt.title('Week wise taxi fares details')

plt.show()
def haversine_distance(lat1, long1, lat2, long2):

    data = [df_train]

    for i in data:

        R = 6371  #radius of earth in kilometers

        #R = 3959 #radius of earth in miles

        phi1 = np.radians(i[lat1])

        phi2 = np.radians(i[lat2])

    

        delta_phi = np.radians(i[lat2]-i[lat1])

        delta_lambda = np.radians(i[long2]-i[long1])

    

        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    

        #c = 2 * atan2( √a, √(1−a) )

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    

        #d = R*c

        d = (R * c) #in kilometers

        i['h_distance'] = d

    return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df_train['h_distance'].describe()
df_train=df_train.drop(df_train[df_train['h_distance']==0].index, axis=0)
df_train['h_distance'].describe()
df_train['fare_amount'].describe()
df_train=df_train.drop(df_train[df_train['fare_amount']==0].index, axis=0)
df_train['fare_amount'].describe()