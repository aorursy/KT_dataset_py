# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
fare = pd.read_csv('../input/sample_submission.csv')
datatest = pd.read_csv('../input/test.csv', nrows =100_000, parse_dates = ['pickup_datetime'])
train = pd.read_csv('../input/train.csv', nrows =100_000, parse_dates = ['pickup_datetime'])


train.dtypes

train.describe()

datatest.describe()
# Remove datas with negative fares and weird values
print('Old size: %d' % len(train))
train = train[train.fare_amount>=0]
train = train[train.pickup_longitude>=-80]
train = train[train.pickup_longitude<=-70]
train = train[train.pickup_latitude>=30]
train = train[train.pickup_latitude<=45]
train = train[train.dropoff_longitude>=-80]
train = train[train.dropoff_longitude<=-70]
train = train[train.dropoff_latitude>=30]
train = train[train.passenger_count>0]

#remove 0 passenger race
train = train[train.dropoff_latitude<=45]

print('New size: %d' % len(train))


# plot fare datagram
train[train.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Fare Value');
#Check if ther is missing data
print(train.isnull().sum())


#drop it
print('Old size: %d' % len(train))
train = train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train))
#new DataSet Statistics
train.describe()

#Distance calculation on a sphere
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

# add new column to dataframe with distance in miles
train['distance_miles'] = distance(train.pickup_latitude, train.pickup_longitude, train.dropoff_latitude, train.dropoff_longitude)
datatest['distance_miles'] = distance(datatest.pickup_latitude, datatest.pickup_longitude, datatest.dropoff_latitude, datatest.dropoff_longitude)

#set max distance 100 miles
train = train[train.distance_miles<100]

#plot graph
train.distance_miles.hist(bins=40, figsize=(11,5))
plt.xlabel('distance miles')
plt.title('Histogram ride distances in miles')
train.distance_miles.describe()
train.groupby('passenger_count')['distance_miles', 'fare_amount'].mean()

#conversion type and check it
train['key'] = pd.to_datetime(train['key'])
train['pickup_datetime']  = pd.to_datetime(train['pickup_datetime'])

train.dtypes
#Create field date 
train['Day'] = train['pickup_datetime'].dt.day
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day of Week'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour

train.head(5)
#Create field date 
datatest['Day'] = datatest['pickup_datetime'].dt.day
datatest['Month'] = datatest['pickup_datetime'].dt.month
datatest['Date'] = datatest['pickup_datetime'].dt.day
datatest['Day of Week'] = datatest['pickup_datetime'].dt.dayofweek
datatest['Hour'] = datatest['pickup_datetime'].dt.hour

datatest.head(5)
#Check day fare amount
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')
#check hour fare amount
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')
#Check day of the week fare amount
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day of Week'], y=train['fare_amount'], s=1.5)
plt.xlabel('Day')
plt.ylabel('Fare')
#Average price/mile
print("Average $USD/Mile : {:0.2f}".format(train.fare_amount.sum()/train.distance_miles.sum()))
# scatter plot distance - fare
fig, axs = plt.subplots(1, 2, figsize=(16,6))
axs[0].scatter(train.distance_miles, train.fare_amount, alpha=0.2)
axs[0].set_xlabel('distance mile')
axs[0].set_ylabel('fare $USD')
axs[0].set_title('All data')

# zoom in on part of data
idx = (train.distance_miles < 15) & (train.fare_amount < 100)
axs[1].scatter(train[idx].distance_miles, train[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('distance mile')
axs[1].set_ylabel('fare $USD')
axs[1].set_title('Zoom in on distance < 15 mile, fare < $100');
#Remove data where mile = 0 and fare = 0
print('Old size: %d' % len(train))
train = train[train.distance_miles>0.05]
train = train[train.fare_amount>0.05]

print('New size: %d' % len(train))

print("Average $USD/Mile : {:0.2f}".format(train.fare_amount.sum()/train.distance_miles.sum()))

#New graph with linear approch
# scatter plot distance - fare
fig, axs = plt.subplots(1, 2,figsize=(16,6))
axs[0].scatter(train.distance_miles, train.fare_amount, alpha=0.2)
axs[0].set_xlabel('distance mile')
axs[0].set_ylabel('fare $USD')
axs[0].set_title('All data')
x = [0,2,4,10,20]
y = [0,10.48, 20.96, 52.4 ,104.8]
axs[0].plot(x,y,color='r')
# zoom in on part of data
idx = (train.distance_miles < 15) & (train.fare_amount < 100)
axs[1].scatter(train[idx].distance_miles, train[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('distance mile')
axs[1].set_ylabel('fare $USD')
axs[1].set_title('Zoom in on distance < 15 mile, fare < $100');
x = [0,2,4,10,20]
y = [0,10.48, 20.96, 52.4 ,104.8]
axs[1].plot(x,y,color='r')
#New linear equation with test datas
datatest.dtypes
train["fare_prediction"] = (distance(train.pickup_latitude, train.pickup_longitude, train.dropoff_latitude, train.dropoff_longitude)*5.24)
train["fare_diff"]= abs(train.fare_amount - train.fare_prediction)
train.head(10)
#remove coloms for Random forest algo
train = train.drop(['key','pickup_datetime'], axis = 1)
train = train.drop(['fare_prediction'], axis = 1)
train = train.drop(['fare_diff'], axis = 1)

datatest = datatest.drop(['key','pickup_datetime'], axis = 1)


#check the data
train.columns

#check the test data
datatest.columns
x_train = train.iloc[:,train.columns!='fare_amount']
y_train = train['fare_amount'].values

x_test = datatest
datatest.head(5)
x_train.shape

y_train.shape

x_test.shape


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = rf_predict
submission.to_csv('submission_1.csv', index=False)
submission.head(20)