# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import datetime as dt

from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))

pd.options.mode.chained_assignment = None

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv", sep=',', lineterminator='\n',nrows=1000000)

df_train.rename(columns={'passenger_count\r':'passenger_count'}, inplace=True)

print("df_train size ",df_train.info()) 



df_test=pd.read_csv("../input/test.csv", sep=',', lineterminator='\n')

df_test.rename(columns={'passenger_count\r':'passenger_count'}, inplace=True)

print("df_test size ",df_test.info()) 



"""

key                  1000000 non-null object

fare_amount          1000000 non-null float64

pickup_datetime      1000000 non-null object

pickup_longitude     1000000 non-null float64

pickup_latitude      1000000 non-null float64

dropoff_longitude    999990 non-null float64

dropoff_latitude     999990 non-null float64

"""
#df_train.describe(include='all')

print(df_train.shape)

print(df_train.nunique())

"""

key                  1000000

fare_amount             2137

pickup_datetime       861755

pickup_longitude      113607

pickup_latitude       144938

dropoff_longitude     134494

dropoff_latitude      171395

passenger_count\r          8

dtype: int64

"""
#scatter graph

#tips = sns.load_dataset("df_train")

sns.relplot(x="passenger_count", y="fare_amount", data=df_train);

# remove data with 0 passengers and upper outliner >200

# have_pass is a boolean variable with True or False in it



have_pass =  (df_train.passenger_count >0) & (df_train.passenger_count <10)

have_money=  df_train['fare_amount']>0

df_train_2 = df_train[have_pass & have_money]

df_test_2=df_test

print("filter completed")
sns.relplot(x="passenger_count", y="fare_amount", data=df_train_2);
# remove upper outliner fares

over_price=  df_train_2['fare_amount']<=300

df_train_2 = df_train_2[over_price]
sns.relplot(x="passenger_count", y="fare_amount", data=df_train_2);
# calculate distance

def  getDistance(lat1,lon1,lat2,lon2):

    R = 6373.0

    

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance

df_train_2["distance"]=getDistance(df_train_2.pickup_latitude,df_train_2.pickup_longitude,df_train_2.dropoff_latitude,df_train_2.dropoff_longitude)

print("train distance completed")



df_test_2["distance"]=getDistance(df_test_2.pickup_latitude,df_test_2.pickup_longitude,df_test_2.dropoff_latitude,df_test_2.dropoff_longitude)

print("test distance completed")





# create column to define day is weekday or weekend    

df_train_2['date_pickup']=pd.to_datetime(df_train_2['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

#df_train_2['WeekValue'] = np.where(df_train_2['date_pickup'].dayofweek<5, 0, 1) 

print("train conversion completed")



df_test_2['date_pickup']=pd.to_datetime(df_test_2['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

#df_train_2['WeekValue'] = np.where(df_train_2['date_pickup'].dayofweek<5, 0, 1) 

print("test conversion completed")







WeekValue_dict = {0:0,1: 0, 2: 0, 3: 0, 4: 0,5: 1, 6: 1}

df_train_2['WeekValue'] = df_train_2['date_pickup'].dt.dayofweek.map(WeekValue_dict)

print("train WeekValue completed")   



df_test_2['WeekValue'] = df_test_2['date_pickup'].dt.dayofweek.map(WeekValue_dict)

print("test WeekValue completed")   









# seperate month - day - time values

df_train_2['year'] = df_train_2['date_pickup'].dt.year

df_train_2['month'] = df_train_2['date_pickup'].dt.month

df_train_2['day'] = df_train_2['date_pickup'].dt.day

df_train_2['hour'] = df_train_2['date_pickup'].dt.hour

print(" train dates completed")





# seperate month - day - time values

df_test_2['year'] = df_test_2['date_pickup'].dt.year

df_test_2['month'] = df_test_2['date_pickup'].dt.month

df_test_2['day'] = df_test_2['date_pickup'].dt.day

df_test_2['hour'] = df_test_2['date_pickup'].dt.hour

print(" test dates completed")

# create  column to define day interval ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

day_int_dict = {6:0,7:0,8:0,9:0,10:0,11:0,12:1,13:1,14:2,15:2,16:2,17:2,18:2,0:4,1:4,2:4,3:4,4:4,5:4,19:3,

20:3,21:3,22:3,23:3}

df_train_2['day_int'] = df_train_2['hour'].map(day_int_dict)

df_train_2['day_name'] = df_train_2['date_pickup'].dt.day_name()

#0 winter ,1 april , 2 summer,3 Autumn

season_dict = {12:0,1: 0, 2: 0, 3: 1, 4: 1,5: 1, 6: 2, 7: 2, 8: 2,9: 3, 10: 3, 11: 3}

df_train_2['season'] = df_train_2['month'].map(season_dict)



print("train 3rd completed")



df_test_2['day_int'] = df_test_2['hour'].map(day_int_dict)

df_test_2['day_name'] = df_test_2['date_pickup'].dt.day_name()

df_test_2['season'] = df_test_2['month'].map(season_dict)



print("test 3rd completed")

sns.barplot(x="WeekValue", y="fare_amount", data=df_train_2, estimator=sum)
# ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

g=sns.lineplot(x="day_int", y="fare_amount",hue='WeekValue',ci=None, data=df_train_2, estimator=np.sum)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
# ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

g=sns.lineplot(x="day_int", y="fare_amount",hue='WeekValue',ci=None,data=df_train_2,estimator=np.mean)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
# ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

g=sns.barplot(x="day_int", y="fare_amount",hue='day_name',hue_order=['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'],ci=None,data=df_train_2,estimator=np.sum)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
#0 winter ,1 april , 2 summer,3 Autumn

g=sns.barplot(x="season", y="fare_amount",hue='day_name',hue_order=['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'],ci=None,data=df_train_2,estimator=np.sum)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)