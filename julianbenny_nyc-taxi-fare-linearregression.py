# Initial Python environment setup...

import numpy as np # linear algebra

import pandas as pd # CSV file I/O (e.g. pd.read_csv)

import os # reading the input files we have access to



print(os.listdir('../input'))
train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)

train_df.dtypes
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes
# Given a dataframe, add two new features 'abs_diff_longitude' and

# 'abs_diff_latitude' reprensenting the "Manhattan vector" from

# the pickup location to the dropoff location.





def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

    

    



add_travel_vector_features(train_df)

add_travel_vector_features(test_df)
train_df.columns
print(train_df.isnull().sum())
print('Old size: %d' % len(train_df))

train_df = train_df.dropna(how = 'any', axis = 0)

print('New size: %d' % len(train_df))
plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(train_df))

train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]

print('New size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
train_df.head()
train_df['pickup_datetime'][0][11:19]
list1 = list(train_df['pickup_datetime'])               # Creating an extra col of pickup time,extracting from pickup_datetime



for i in range(len(list1)):

    list1[i] = list1[i][11:19]



train_df['pickup_time'] = list1







list2 = list(test_df['pickup_datetime'])



for i in range(len(list2)):

    list2[i] = list2[i][11:19]



test_df['pickup_time'] = list2

    
train_df.head()
test_df.head()
x=pd.Timestamp(train_df['pickup_datetime'][0][:-4]).dayofweek

x
# Creating an extra col for day of the week



list1 = list(train_df['pickup_datetime'])



for i in range(len(list1)):

    list1[i] = pd.Timestamp(list1[i][:-4]).dayofweek



train_df['weekday'] = list1





list2 = list(test_df['pickup_datetime'])



for i in range(len(list2)):

    list2[i] = pd.Timestamp(list2[i][:-4]).dayofweek



test_df['weekday'] = list2

test_df.head()
# Dropping "pickup_datetime" col



train_df.drop("pickup_datetime",axis=1,inplace=True)

test_df.drop("pickup_datetime",axis=1,inplace=True)
test_df.head()
train_df.head()
col = test_df.columns.tolist()

col = col[:6] + col[8:] +col[6:8]

col



test_df = test_df[col]

test_df.head()
train_df.shape
train_df['weekday'].replace(to_replace=[i for i in range(0,7)],

                           value=["monday","tuesday",'wednesday','thursday','friday','saturday','sunday'],

                           inplace=True)



test_df['weekday'].replace(to_replace=[i for i in range(0,7)],

                           value=["monday","tuesday",'wednesday','thursday','friday','saturday','sunday'],

                           inplace=True)
train_df.head()
test_df.head()
train_one_hot = pd.get_dummies(train_df['weekday'])

train_df = pd.concat([train_df,train_one_hot],axis=1)



test_one_hot = pd.get_dummies(test_df['weekday'])

test_df = pd.concat([test_df,test_one_hot],axis=1)

test_df.head()
train_df.head()
train_df.drop("weekday",axis=1,inplace=True)

test_df.drop("weekday",axis=1,inplace=True)
a = train_df['pickup_time'][0].split(":")

(int(a[0])*100) + int(a[1]) + float(a[2])/100

# Converting pickup_time to float



list1 = list(train_df['pickup_time'])

for i in range(len(list1)):

    a = list1[i].split(":")

    list1[i] = (int(a[0])*100) + int(a[1]) + float(a[2])/100



train_df['pickup_time'] = list1



list2 = list(test_df['pickup_time'])

for i in range(len(list2)):

    a = list2[i].split(":")

    list2[i] = (int(a[0])*100) + int(a[1]) + float(a[2])/100



test_df['pickup_time'] = list2

train_df.head()
test_df.head()
# rearranging cols

test_df = test_df[train_df.drop('fare_amount',axis=1).columns]
test_df.head()
train_df.head()
# Calculating distance in kms



R = 6373.0

lat1 =np.asarray(np.radians(train_df['pickup_latitude']))

lon1 = np.asarray(np.radians(train_df['pickup_longitude']))

lat2 = np.asarray(np.radians(train_df['dropoff_latitude']))

lon2 = np.asarray(np.radians(train_df['dropoff_longitude']))



dlon = lon2 - lon1

dlat = lat2 - lat1

ls1=[] 

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distance = R * c



    

train_df['Distance']=np.asarray(distance)*0.621







lat1 =np.asarray(np.radians(test_df['pickup_latitude']))

lon1 = np.asarray(np.radians(test_df['pickup_longitude']))

lat2 = np.asarray(np.radians(test_df['dropoff_latitude']))

lon2 = np.asarray(np.radians(test_df['dropoff_longitude']))



dlon = lon2 - lon1

dlat = lat2 - lat1

 

a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distance = R * c

test_df['Distance']=np.asarray(distance)*0.621
train_df.head()
test_df.head()
# Calculated distances in ref to the airport



R = 6373.0

lat1 =np.asarray(np.radians(train_df['pickup_latitude']))

lon1 = np.asarray(np.radians(train_df['pickup_longitude']))

lat2 = np.asarray(np.radians(train_df['dropoff_latitude']))

lon2 = np.asarray(np.radians(train_df['dropoff_longitude']))



lat3=np.zeros(len(train_df))+np.radians(40.6413111)

lon3=np.zeros(len(train_df))+np.radians(-73.7781391)

dlon_pickup = lon3 - lon1

dlat_pickup = lat3 - lat1

d_lon_dropoff=lon3 -lon2

d_lat_dropoff=lat3-lat2

a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2

c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))

distance1 = R * c1

train_df['Pickup_Distance_airport']=np.asarray(distance1)*0.621



a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2

c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))

distance2 = R * c2



    

train_df['Dropoff_Distance_airport']=np.asarray(distance2)*0.621







lat1 =np.asarray(np.radians(test_df['pickup_latitude']))

lon1 = np.asarray(np.radians(test_df['pickup_longitude']))

lat2 = np.asarray(np.radians(test_df['dropoff_latitude']))

lon2 = np.asarray(np.radians(test_df['dropoff_longitude']))



lat3=np.zeros(len(test_df))+np.radians(40.6413111)

lon3=np.zeros(len(test_df))+np.radians(-73.7781391)

dlon_pickup = lon3 - lon1

dlat_pickup = lat3 - lat1

d_lon_dropoff=lon3 -lon2

d_lat_dropoff=lat3-lat2

a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2

c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))

distance1 = R * c1

test_df['Pickup_Distance_airport']=np.asarray(distance1)*0.621



a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2

c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))

distance2 = R * c2



test_df['Dropoff_Distance_airport']=np.asarray(distance2)*0.621

# Rounding off data to two decimal places



train_df['Distance']=np.round(train_df['Distance'],2)

train_df['Pickup_Distance_airport']=np.round(train_df['Pickup_Distance_airport'],2)

train_df['Dropoff_Distance_airport']=np.round(train_df['Dropoff_Distance_airport'],2)



test_df['Distance']=np.round(test_df['Distance'],2)

test_df['Pickup_Distance_airport']=np.round(test_df['Pickup_Distance_airport'],2)

test_df['Dropoff_Distance_airport']=np.round(test_df['Dropoff_Distance_airport'],2)
train_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)

test_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
train_df.head()
test_df.head()
print(train_df.shape , test_df.shape)
from sklearn.model_selection import train_test_split



X=train_df.drop(['key','fare_amount'],axis=1)

y=train_df['fare_amount']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=80)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train,y_train)

reg.score(X_test,y_test)
predictions = reg.predict(test_df.drop("key",axis=1))

predictions = np.round(predictions,2)

predictions
Submission=pd.DataFrame(data=predictions,columns=['fare_amount'])



Submission['key']=test_df['key']



Submission=Submission[['key','fare_amount']]
Submission.set_index('key',inplace=True)
Submission.reset_index().head()
Submission.to_csv('Submission.csv')