# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows=10_000_000)

train_data.head()
train_data.shape
train_data.info()
test_data=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')

test_data.head()
test_data.info()
train_data.isna().sum()
train_data['Difference_longitude']=np.abs(np.asarray(train_data['pickup_longitude']-train_data['dropoff_longitude']))

train_data['Difference_latitude']=np.abs(np.asarray(train_data['pickup_latitude']-train_data['dropoff_latitude']))





test_data['Difference_longitude']=np.abs(np.asarray(test_data['pickup_longitude']-test_data['dropoff_longitude']))

test_data['Difference_latitude']=np.abs(np.asarray(test_data['pickup_latitude']-test_data['dropoff_latitude']))
print(f'Before Dropping null values: {len(train_data)}')

train_data.dropna(inplace=True)

print(f'After Dropping null values: {len(train_data)}')
plot = train_data[:2000].plot.scatter('Difference_longitude', 'Difference_latitude')
train_data=train_data[(train_data['Difference_longitude']<5.0)&(train_data['Difference_latitude']<5.0)]
ls1=list(train_data['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][11:-7:]

train_data['pickuptime']=ls1    







ls1=list(test_data['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][11:-7:]

test_data['pickuptime']=ls1   
train_data.head()
ls1=list(train_data['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][:-4:]

    ls1[i]=pd.Timestamp(ls1[i])

    ls1[i]=ls1[i].weekday()

train_data['Weekday']=ls1





ls1=list(test_data['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][:-4:]

    ls1[i]=pd.Timestamp(ls1[i])

    ls1[i]=ls1[i].weekday()

test_data['Weekday']=ls1
train_data.head()
test_data.head()
train_data.drop('pickup_datetime',inplace=True,axis=1)

test_data.drop('pickup_datetime',inplace=True,axis=1)
train_data['Weekday'].replace(to_replace=[i for i in range(0,7)],

                            value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],

                              inplace=True)

test_data['Weekday'].replace(to_replace=[i for i in range(0,7)],

                              value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],

                              inplace=True)
train_one_hot=pd.get_dummies(train_data['Weekday'])

test_one_hot=pd.get_dummies(test_data['Weekday'])

train_data=pd.concat([train_data,train_one_hot],axis=1)

test_data=pd.concat([test_data,test_one_hot],axis=1)
train_data.drop('Weekday',axis=1,inplace=True)

test_data.drop('Weekday',axis=1,inplace=True)
ls1=list(train_data['pickuptime'])

for i in range(len(ls1)):

    z=ls1[i].split(':')

    ls1[i]=int(z[0])*100+int(z[1])

train_data['pickuptime']=ls1





ls1=list(test_data['pickuptime'])

for i in range(len(ls1)):

    z=ls1[i].split(':')

    ls1[i]=int(z[0])*100+int(z[1])

test_data['pickuptime']=ls1
train_data.head()
R = 6373.0

lat1 =np.asarray(np.radians(train_data['pickup_latitude']))

lon1 = np.asarray(np.radians(train_data['pickup_longitude']))

lat2 = np.asarray(np.radians(train_data['dropoff_latitude']))

lon2 = np.asarray(np.radians(train_data['dropoff_longitude']))



dlon = lon2 - lon1

dlat = lat2 - lat1

ls1=[] 

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distance = R * c



    

train_data['Distance']=np.asarray(distance)*0.621







lat1 =np.asarray(np.radians(test_data['pickup_latitude']))

lon1 = np.asarray(np.radians(test_data['pickup_longitude']))

lat2 = np.asarray(np.radians(test_data['dropoff_latitude']))

lon2 = np.asarray(np.radians(test_data['dropoff_longitude']))



dlon = lon2 - lon1

dlat = lat2 - lat1

 

a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distance = R * c

test_data['Distance']=np.asarray(distance)*0.621
R = 6373.0

lat1 =np.asarray(np.radians(train_data['pickup_latitude']))

lon1 = np.asarray(np.radians(train_data['pickup_longitude']))

lat2 = np.asarray(np.radians(train_data['dropoff_latitude']))

lon2 = np.asarray(np.radians(train_data['dropoff_longitude']))



lat3=np.zeros(len(train_data))+np.radians(40.6413111)

lon3=np.zeros(len(train_data))+np.radians(-73.7781391)

dlon_pickup = lon3 - lon1

dlat_pickup = lat3 - lat1

d_lon_dropoff=lon3 -lon2

d_lat_dropoff=lat3-lat2

a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2

c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))

distance1 = R * c1

train_data['Pickup_Distance_airport']=np.asarray(distance1)*0.621



a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2

c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))

distance2 = R * c2



    

train_data['Dropoff_Distance_airport']=np.asarray(distance2)*0.621







lat1 =np.asarray(np.radians(test_data['pickup_latitude']))

lon1 = np.asarray(np.radians(test_data['pickup_longitude']))

lat2 = np.asarray(np.radians(test_data['dropoff_latitude']))

lon2 = np.asarray(np.radians(test_data['dropoff_longitude']))



lat3=np.zeros(len(test_data))+np.radians(40.6413111)

lon3=np.zeros(len(test_data))+np.radians(-73.7781391)

dlon_pickup = lon3 - lon1

dlat_pickup = lat3 - lat1

d_lon_dropoff=lon3 -lon2

d_lat_dropoff=lat3-lat2

a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2

c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))

distance1 = R * c1

test_data['Pickup_Distance_airport']=np.asarray(distance1)*0.621



a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2

c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))

distance2 = R * c2



    

test_data['Dropoff_Distance_airport']=np.asarray(distance2)*0.621
train_data['Distance']=np.round(train_data['Distance'],2)

train_data['Pickup_Distance_airport']=np.round(train_data['Pickup_Distance_airport'],2)

train_data['Dropoff_Distance_airport']=np.round(train_data['Dropoff_Distance_airport'],2)

test_data['Distance']=np.round(test_data['Distance'],2)

test_data['Pickup_Distance_airport']=np.round(test_data['Pickup_Distance_airport'],2)

test_data['Dropoff_Distance_airport']=np.round(test_data['Dropoff_Distance_airport'],2)
train_data.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)

test_data.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
train_data['Difference_longitude']=np.abs(train_data['Difference_longitude']-np.mean(train_data['Difference_longitude']))

train_data['Difference_longitude']=train_data['Difference_longitude']/np.var(train_data['Difference_longitude'])
train_data['Difference_latitude']=np.abs(train_data['Difference_latitude']-np.mean(train_data['Difference_latitude']))

train_data['Difference_latitude']=train_data['Difference_latitude']/np.var(train_data['Difference_latitude'])
test_data['Difference_longitude']=np.abs(test_data['Difference_longitude']-np.mean(test_data['Difference_longitude']))

test_data['Difference_longitude']=test_data['Difference_longitude']/np.var(test_data['Difference_longitude'])



test_data['Difference_latitude']=np.abs(test_data['Difference_latitude']-np.mean(test_data['Difference_latitude']))

test_data['Difference_latitude']=test_data['Difference_latitude']/np.var(test_data['Difference_latitude'])
train_data.shape
test_data.shape
from sklearn.model_selection import train_test_split

X=train_data.drop(['key','fare_amount'],axis=1)

y=train_data['fare_amount']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=80)
from sklearn.linear_model import LinearRegression

lr=LinearRegression(normalize=True)

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))
pred=np.round(lr.predict(test_data.drop('key',axis=1)),2)
pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv').head()
Submission=pd.DataFrame(data=pred,columns=['fare_amount'])

Submission['key']=test_data['key']

Submission=Submission[['key','fare_amount']]
Submission.set_index('key',inplace=True)
Submission.to_csv('Submission.csv')