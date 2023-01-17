# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import geopy.distance
#from geopy.distance import geodesic
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', nrows = 1000000)
test = pd.read_csv('../input/test.csv')
# Checking the shape of train datasets
train.shape
# Check the shape of test datasets
test.shape
train.isnull().sum()
test.isnull().sum()
train.describe()
# drop the null value contain in row...
train = train.drop(train[train.isnull().any(1)].index, axis = 0)
train.head()
# Visualize the data['fare_amount'] columns
train[train.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');
# Check distribution of taxi fare
train['fare_amount'].describe()
# Null value in fare_amount columns and count them
train[train['fare_amount']<0]['fare_amount'].count()
train = train.drop(train[train['fare_amount']<0].index, axis = 0)
train.shape
#no more negative values in the fare field
train['fare_amount'].describe()
# Though from the graph we see distribution is Negative Skew, but among distribution what is the highest amount.
train['fare_amount'].sort_values(ascending=False).head()
train['passenger_count'].describe()
# A taxi cannot carry 208 value more than 6, lets check how many rows are there rows are more then 6 Passenger in single taxi
train[train['passenger_count']>6]['passenger_count'].count()
train = train.drop(train[train['passenger_count']>6].index, axis = 0)
train.shape
train['passenger_count'].describe()
# Visualize the data['fare_amount'] columns
sns.barplot(x = train['passenger_count'],y = train['fare_amount'], data = train)
plt.xlabel('Passenger Count')
plt.ylabel('Fare Amount')
plt.title('Histogram');
# Visualize the data['fare_amount'] columns
train[train.passenger_count>0].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');
train[train.passenger_count==0]['passenger_count'].count()#['fare_amount'].sum()#['passenger_count']
train[train.passenger_count==0]
train = train.drop(train[train['passenger_count']==0].index, axis = 0)
train.shape
# Explore the pickup latitude and longitudes
train['pickup_latitude'].describe()
train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]
# Drop the unwanted rows
train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)
train.shape
train['pickup_longitude'].describe()
train[train['pickup_longitude']<-180]
train[train['pickup_longitude']>180]
train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)
#11 rows dropped
train.shape
# We have to same procedure for dropoff latitude and longitude
train[train['dropoff_latitude']<-90]
train[train['dropoff_latitude']>90]
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)
#8 rows dropped
train.shape
train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]
train.dtypes
train['key'] = pd.to_datetime(train.key, format='%Y-%m-%d %H:%M')
train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
test['key'] = pd.to_datetime(train.key, format='%Y-%m-%d %H:%M:%S UTC')
test['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
train.head()
train.dtypes
train[(train.pickup_latitude==0) | (train.pickup_longitude)==0 | (train.dropoff_latitude==0)|(train.dropoff_longitude==0)]
print("Range of Pickup Latitude is ", (min(train['pickup_latitude']),max(train['pickup_latitude'])))
train[train['pickup_longitude'] == 0]
from geopy.distance import geodesic

def pandasVincenty(df):
    '''calculate distance between two lat&long points using the Vincenty formula '''

    return geodesic((df.pickup_latitude, df.pickup_longitude), (df.dropoff_latitude, df.dropoff_longitude)).km


train['distance_kms'] =  train.apply(lambda r: pandasVincenty(r), axis=1)
train.head(10)
train[(train['pickup_longitude'] == 0) | (train['pickup_latitude'] == 0) | (train['dropoff_longitude'] == 0) | (train['dropoff_latitude'] == 0)]
# Checking the number of ride for each passenger_count. 
train['passenger_count'].value_counts().plot(kind='bar')
plt.xlabel('distribution for passenger_count for taxi ride')
# Creating feature separatly as per houly, date, day of week, month

# For train data
train['Year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day of Week'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour

# Test data
test['Year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day of Week'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
train.head()
plt.figure(figsize=(15,7))
plt.hist(train['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.figure(figsize=(15, 8))
plt.scatter(x = train['Date'], y = train['fare_amount'], s = 1.5)
plt.xlabel('date')
plt.ylabel('fare')
plt.figure(figsize=(15, 8))
plt.hist(x = train['Hour'], bins = 100)
plt.xlabel('hour')
plt.ylabel('fare')
plt.figure(figsize=(15, 6))
sns.violinplot( x=train["Hour"], y=train["fare_amount"], linewidth=0.4)
plt.xlabel('Hour')
train.head()
# import folium package
import folium
 
mark = folium.Map(location = [40.721319, -74.016587],zoom_start = 12)
#folium.Marker(location = [40.721319, -74.016587]).add_to(mark)
#mark
for i in range(0,len(train.head(500))):
    folium.Marker([train.iloc[i]['pickup_latitude'], train.iloc[i]['pickup_longitude']]).add_to(mark)
mark
for i in range(0,len(train.head(500))):
    folium.Marker([train.iloc[i]['dropoff_latitude'], train.iloc[i]['dropoff_longitude']]).add_to(mark)
mark


