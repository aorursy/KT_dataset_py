import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

train_df = pd.read_csv("../input/train.csv",nrows = 200000)
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df.describe()
plt.figure(figsize = (10,6))
sns.distplot(train_df['fare_amount']);
plt.title('Distribution of fare')
#removing values below 0. Certainly an wrong entry.. may be cabbi has broken some rule and hence fined.
#whatever it is we need to remove them.
train_df[(train_df['fare_amount']<0)].shape[0]#13 such entries

train_df = train_df[(train_df['fare_amount']>=0)]
train_df = train_df[(train_df['fare_amount']<=100)]
train_df['fare_amount'].size 

plt.figure(figsize = (10,6))
sns.distplot(train_df['pickup_latitude']);
plt.title('Distribution of latitude')

#40.7128° N  <-- New York City latitude
train_df['pickup_latitude'].mean() #mean of the data we have. (kinda accurate) Fact: 1 degree latitude is 69 miles so all the values other than 38-42 are outliers
plt.figure(figsize = (9,4))
sns.distplot(train_df['pickup_latitude'])
plt.xlim(0,80)
plt.title('Distribution of latitude in more detail')
#There is lot of values between 10-20. I dont know why
cnt = []
r = [35,36,37,38,39,40,41,42,43,44,45]
for i in r:
    no = train_df[(train_df['pickup_latitude']<i)].shape[0]
    cnt.append(no)
plt.scatter(r,cnt) #see the jump
train_df = train_df[(train_df['pickup_latitude']>0)]
train_df = train_df[(train_df['pickup_latitude']>39)] #-->111 #there are only 20 <39 and 111 greater than 4 so removing them
train_df = train_df[(train_df['pickup_latitude']<42)]
#we can see mean has increased
train_df['pickup_latitude'].mean() 
#longitude
#NYC longitude -> 74.0060° W
train_df['pickup_longitude'].mean() #-->-73.96832831445111
plt.figure(figsize = (10,6))
sns.distplot(train_df['pickup_longitude']);
plt.title('Distribution of longitude') #There are outliers nee to remove them
train_df = train_df[(train_df['pickup_longitude']>-75)]
train_df = train_df[(train_df['pickup_longitude']<-72)]
train_df['dropoff_longitude'].mean() #-->-73.90602772217527
train_df['dropoff_latitude'].mean() #--> 40.72183614573583
train_df.shape[0]
train_df = train_df[(train_df['pickup_longitude']>-75)]
train_df = train_df[(train_df['pickup_longitude']<-72)]
train_df['dropoff_longitude'].mean() #->-73.90602772217527
train_df = train_df[(train_df['pickup_latitude']>0)]
train_df = train_df[(train_df['pickup_latitude']>39)] #-->111 #there are only 20 <39 and 111 greater than 4 so removing them
train_df = train_df[(train_df['pickup_latitude']<42)]
train_df['dropoff_latitude'].mean()  #->40.72183614573583
train_df['passenger_count'].describe()

plt.figure(figsize = (10,6))
sns.distplot(train_df['passenger_count']);
plt.title('passenger_count') #At this moment I dont think there is any need to preprocess this data
train_df.describe()

def distance(pickup1,pickup2,destination1,destination2):
    lat1, lon1 = pickup1,pickup2
    lat2, lon2 = destination1,destination2
    radius = 3959 # miles

    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d
a = train_df['pickup_latitude'].values
b = train_df['pickup_longitude'].values
c = train_df['dropoff_latitude'].values
d = train_df['dropoff_longitude'].values
dist = distance(a,b,c,d)
dist[0:20]

dist = pd.Series(dist)

train_df['distance'] = dist
train_df.describe()
train_df[train_df['distance']>500]
dist3 = distance(40.762892,-73.976545,40.76406,-73.79317)
dist3
plt.figure(figsize = (10,6))
sns.distplot(train_df['distance']>100);
plt.title('passenger_count') 
def distance1(pickup1,pickup2,destination1,destination2):
    a = (destination1 - pickup1)*69
    b = (destination2 - pickup2)*69
    d = a**2+b**2
    c = np.sqrt(d)
    return c
dist1 = distance1(a,b,c,d)
dist1 = pd.Series(dist1)
dist1.max()
plt.figure(figsize = (10,6))
sns.kdeplot(train_df['pickup_latitude'])
sns.kdeplot(train_df['pickup_longitude']);
plt.title('Distribution of pickup lat and long')
plt.figure(figsize = (10,6))
sns.kdeplot(train_df['dropoff_latitude']);
sns.kdeplot(train_df['dropoff_longitude']);
plt.title('Distribution of dropoff lat and long')

