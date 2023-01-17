# Initial Python environment setup...

import numpy as np # linear algebra

import pandas as pd # CSV file I/O (e.g. pd.read_csv)

import os # reading the input files we have access to



print(os.listdir('../input'))
train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)

train_df.dtypes
# Given a dataframe, add two new features 'abs_diff_longitude' and

# 'abs_diff_latitude' reprensenting the "Manhattan vector" from

# the pickup location to the dropoff location.

def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()



add_travel_vector_features(train_df)
print(train_df.isnull().sum())
print('Old size: %d' % len(train_df))

train_df = train_df.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(train_df))
plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(train_df))

train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]

print('New size: %d' % len(train_df))
ls1=list(train_df['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][11:-7:]

train_df['pickup_time']=ls1
train_df['pickup_time'].head(5)
test_df=pd.read_csv("../input/test.csv")

test_df.head()
add_travel_vector_features(test_df)
test_df.shape
ls1=list(test_df['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][11:-7:]

test_df['pickup_time']=ls1
ls1=list(test_df['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][:-4:]

    ls1[i]=pd.Timestamp(ls1[i])

    ls1[i]=ls1[i].weekday()

test_df['weekday']=ls1
ls1=list(train_df['pickup_datetime'])

for i in range(len(ls1)):

    ls1[i]=ls1[i][:-4:]

    ls1[i]=pd.Timestamp(ls1[i])

    ls1[i]=ls1[i].weekday()

train_df['weekday']=ls1
train_df.shape
train_df.drop('pickup_datetime',inplace=True,axis=1)
test_df.drop('pickup_datetime',inplace=True,axis=1)
train_df['weekday'].replace(to_replace=[i for i in range(0,7)],value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)
test_df['weekday'].replace(to_replace=[i for i in range(0,7)],value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)
train_one_hot=pd.get_dummies(train_df['weekday'])

train_df=pd.concat([train_df,train_one_hot],axis=1)
test_one_hot=pd.get_dummies(test_df['weekday'])

test_df=pd.concat([test_df,test_one_hot],axis=1)
train_df.drop('weekday',inplace=True,axis=1)
test_df.drop('weekday',inplace=True,axis=1)
ls1=list(train_df['pickup_time'])

for i in range(len(ls1)):

    z=ls1[i].split(':')

    ls1[i]=int(z[0])*100 + int(z[1])

train_df['pickup_time']=ls1
ls1=list(test_df['pickup_time'])

for i in range(len(ls1)):

    z=ls1[i].split(':')

    ls1[i]=int(z[0])*100 + int(z[1])

test_df['pickup_time']=ls1
train_df.shape
# R=6373.0

# lat1=np.asarray(np.radians((train_df['pickup_latitude']).astype(int)))

# lon1=np.asarray(np.radians((train_df['pickup_longitude']).astype(int)))

# lat2=np.asarray(np.radians((train_df['dropoff_latitude']).astype(int)))

# lon2=np.asarray(np.radians((train_df['dropoff_longitude']).astype(int)))

# dlon=lon2-lon1

# dlat=lat2-lat1

# a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

# c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

# distance=R*c

# train_df['Distance']=np.asarray(distance)*0.621

R=6373.0

lat1=np.asarray(np.radians(train_df['pickup_latitude']))

lon1=np.asarray(np.radians(train_df['pickup_longitude']))

lat2=np.asarray(np.radians(train_df['dropoff_latitude']))

lon2=np.asarray(np.radians(train_df['dropoff_longitude']))

dlon=lon2-lon1

dlat=lat2-lat1

a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

distance=R*c

train_df['Distance']=np.asarray(distance)*0.621
# R=6373.0

# lat1=np.asarray(np.radians((test_df['pickup_latitude']).astype(int)))

# lon1=np.asarray(np.radians((test_df['pickup_longitude']).astype(int)))

# lat2=np.asarray(np.radians((test_df['dropoff_latitude']).astype(int)))

# lon2=np.asarray(np.radians((test_df['dropoff_longitude']).astype(int)))

# dlon=lon2-lon1

# dlat=lat2-lat1

# a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

# c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

# distance=R*c

# test_df['Distance']=np.asarray(distance)*0.621

R=6373.0

lat1=np.asarray(np.radians(test_df['pickup_latitude']))

lon1=np.asarray(np.radians(test_df['pickup_longitude']))

lat2=np.asarray(np.radians(test_df['dropoff_latitude']))

lon2=np.asarray(np.radians(test_df['dropoff_longitude']))

dlon=lon2-lon1

dlat=lat2-lat1

a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

distance=R*c

test_df['Distance']=np.asarray(distance)*0.621
train_df['Distance']=np.round(train_df['Distance'],2)

test_df['Distance']=np.round(test_df['Distance'],2)
train_df.shape
train_df['abs_diff_longitude']=np.abs(train_df['abs_diff_longitude'] - np.mean(train_df['abs_diff_longitude']))

train_df['abs_diff_latitude']=np.abs(train_df['abs_diff_latitude'] - np.mean(train_df['abs_diff_latitude']))
test_df['abs_diff_longitude']=np.abs(test_df['abs_diff_longitude'] - np.mean(test_df['abs_diff_longitude']))

test_df['abs_diff_latitude']=np.abs(test_df['abs_diff_latitude'] - np.mean(test_df['abs_diff_latitude']))
train_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],inplace=True,axis=1)
test_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],inplace=True,axis=1)
train_df.head()
test_df.shape
from sklearn.model_selection import train_test_split
X=train_df.drop(['key','fare_amount'],axis=1)

y=train_df['fare_amount']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=80)
X_train.shape
from sklearn.linear_model import LinearRegression

lr=LinearRegression(normalize=True)

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))
pred=np.round(lr.predict(test_df.drop(['key'],axis=1)),2)

submission=pd.DataFrame(data=pred,columns=['fare_amount'])
submission["key"]=test_df["key"]
submission.set_index('key',inplace=True)
submission.to_csv("submission.csv")
submission.head()