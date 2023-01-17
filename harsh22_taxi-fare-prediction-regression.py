# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", nrows = 1000000)

test = pd.read_csv("../input/test.csv")
train.shape

test.shape
train.head(10)
train.describe()
#check for missing values in train data

train.isnull().sum().sort_values(ascending=False)
#drop the missing values

train = train.drop(train[train.isnull().any(1)].index, axis = 0)
train.shape
#check the target column

train['fare_amount'].describe()
#38 fields have negative fare_amount values.

from collections import Counter

Counter(train['fare_amount']<0)
train = train.drop(train[train['fare_amount']<0].index, axis=0)

train.shape
#no more negative values in the fare field

train['fare_amount'].describe()
#highest fare is $500

train['fare_amount'].sort_values(ascending=False)
train['passenger_count'].describe()
#max is 208 passengers. Assuming that a bus is a 'taxi' in NYC, I don't think a bus can carry 208 this is DEFINITELY an outlier. 

#Lets drop it 

train[train['passenger_count']>8]
train = train.drop(train[train['passenger_count']==208].index, axis = 0)
#much neater now! Max number of passengers are 6. Which makes sense is the cab is an SUV :)

train['passenger_count'].describe()
#Next, let us explore the pickup latitude and longitudes

train['pickup_latitude'].describe()
train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]
#We need to drop these outliers

train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)
train.shape
#similar operation for pickup longitude

train['pickup_longitude'].describe()
train[train['pickup_longitude']<-180]
train[train['pickup_longitude']>180]
train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)
#similar operation for dropoff latitude and longitude

train[train['dropoff_latitude']<-90]
train[train['dropoff_latitude']>90]
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)
train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]
train.dtypes
train['key'] = pd.to_datetime(train['key'],infer_datetime_format = True)

train['pickup_datetime']  =  pd.to_datetime(train['pickup_datetime'],infer_datetime_format=True)
#Convert for test data

test['key'] = pd.to_datetime(test['key'],infer_datetime_format = True)

test['pickup_datetime']  = pd.to_datetime(test['pickup_datetime'],infer_datetime_format = True)
#check the dtypes after conversion

train.dtypes
def haversine_distance(lat1,long1, lat2,long2):

    data = [train,test]

    for i in data:

        r = 6371

        phi1 = np.radians(i[lat1])

        phi2 = np.radians(i[lat2])

        

        delta_phi = np.radians(i[lat2]-i[lat1])

        delta_lambda = np.radians(i[long2]-i[long1])

        

        a = np.sin(delta_phi /2.0)**2 + np.cos(phi1)* np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    

        

        d = (r * c) #in kilometers

        i['H_Distance'] = d

    return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
train['H_Distance'].head(10)
data = [train,test]

for i in data:

    i['Year'] = i['pickup_datetime'].dt.year

    i['Month'] = i['pickup_datetime'].dt.month

    i['Date'] = i['pickup_datetime'].dt.day

    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek

    i['Hour'] = i['pickup_datetime'].dt.hour
#pickup latitude and longitude = 0

train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)]
train = train.drop(train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)].index, axis=0)
#1 row dropped

train.shape
#dropoff latitude and longitude = 0

train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)]
train = train.drop(train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)].index, axis=0)
high_distance = train.loc[(train['H_Distance']>200)&(train['fare_amount']!=0)]
high_distance
high_distance['H_Distance'] = high_distance.apply(

    lambda row: (row['fare_amount'] - 2.50)/1.56,

    axis=1

)
#sync the train data with the newly computed distance values from high_distance dataframe

train.update(high_distance)
train[train['H_Distance']==0]
train[(train['H_Distance']==0)&(train['fare_amount']==0)]
train = train.drop(train[(train['H_Distance']==0)&(train['fare_amount']==0)].index, axis = 0)
#4 rows dropped

train[(train['H_Distance']==0)].shape
#Between 6AM and 8PM on Mon-Fri

rush_hour = train.loc[(((train['Hour']>=6)&(train['Hour']<=20)) & ((train['Day of Week']>=1) & (train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 2.5))]

rush_hour
train=train.drop(rush_hour.index, axis=0)
#Between 8PM and 6AM on Mon-Fri

non_rush_hour = train.loc[(((train['Hour']<6)|(train['Hour']>20)) & ((train['Day of Week']>=1)&(train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0))]

#print(Counter(non_work_hours['Hour']))

#print(Counter(non_work_hours['Day of Week']))

non_rush_hour

#keep these. Since the fare_amount is not <2.5 (which is the base fare), these values seem legit to me.
#Saturday and Sunday all hours

weekends = train.loc[((train['Day of Week']==0) | (train['Day of Week']==6)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0)]

weekends

#Counter(weekends['Day of Week'])

#keep these too. Since the fare_amount is not <2.5, these values seem legit to me.
train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]
scenario_3 = train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]
len(scenario_3)
#We do not have any distance values that are outliers.

scenario_3.sort_values('H_Distance', ascending=False)
scenario_3['fare_amount'] = scenario_3.apply(

    lambda row: ((row['H_Distance'] * 1.56) + 2.50), axis=1

)
scenario_3['fare_amount']
train.update(scenario_3)
train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]
scenario_4 = train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]
len(scenario_4)
#Using our prior knowledge about the base price during weekdays and weekends for the cabs.

#I do not want to impute these 1502 values as they are legible ones.

scenario_4.loc[(scenario_4['fare_amount']<=3.0)&(scenario_4['H_Distance']==0)]
scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]
scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]
scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]
len(scenario_4_sub)
scenario_4_sub['H_Distance'] = scenario_4_sub.apply(

lambda row: ((row['fare_amount']-2.50)/1.56), axis=1

)
train.update(scenario_4_sub)
train.columns
test.columns
#not including the pickup_datetime columns as datetime columns cannot be directly used while modelling. Features need to extracted from the 

#timestamp fields which will later be used as features for modelling.

train = train.drop(['key','pickup_datetime'], axis = 1)

test = test.drop(['key','pickup_datetime'], axis = 1)
x_train = train.iloc[:,train.columns!='fare_amount']

y_train = train['fare_amount'].values

x_test = test
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(x_train, y_train)

rf_predict = rf.predict(x_test)

#print(rf_predict)
submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = rf_predict

submission.to_csv('submission_1.csv', index=False)

submission.head(20)
from sklearn import linear_model

lr = linear_model.LinearRegression()

lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = lr_predict

submission.to_csv('submission_2.csv', index=False)

submission.head(20)