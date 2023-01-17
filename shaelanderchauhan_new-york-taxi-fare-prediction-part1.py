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
import numpy as np

import pandas as pd
train = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows = 3000000)

test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
train.shape
test.shape
train.head(10)
train.describe()
#check for missing values in train data

train.isnull().sum().sort_values(ascending=False)


test.isnull().sum().sort_values(ascending=False)
train = train.dropna(subset=['dropoff_latitude'])
train.isnull().sum().sort_values(ascending=False)
train['fare_amount'].describe()
#38 fields have negative fare_amount values.

from collections import Counter

Counter(train['fare_amount']<0)

train = train.drop(train[train['fare_amount']<0].index, axis=0)
#no more negative values in the fare field

train['fare_amount'].describe()
#highest fare is $500

train['fare_amount'].sort_values(ascending=False)
train.describe()
train['passenger_count'].describe()
train[train['passenger_count']>6]
train = train.drop(train[train['passenger_count']>6].index, axis = 0)
train[train['passenger_count']>6]
#much neater now! Max number of passengers are 6. Which makes sense is the cab is an SUV :)

train['passenger_count'].sort_values(ascending=False)
test['passenger_count'].sort_values(ascending= False)
print(f'Rows before removing coordinate outliers - {train.shape[0]}')



train = train[train.pickup_longitude.between(test.pickup_longitude.min(), test.pickup_longitude.max())]

train = train[train.pickup_latitude.between(test.pickup_latitude.min(), test.pickup_latitude.max())]

train = train[train.dropoff_longitude.between(test.dropoff_longitude.min(), test.dropoff_longitude.max())]

train = train[train.dropoff_latitude.between(test.dropoff_latitude.min(), test.dropoff_latitude.max())]



print(f'Rows after removing coordinate outliers - {train.shape[0]}')
train.describe()
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    """

    Return distance along great radius between pickup and dropoff coordinates.

    """

    #Define earth radius (km)

    R_earth = 6371

    #Convert degrees to radians

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,

                                                             [pickup_lat, pickup_lon, 

                                                              dropoff_lat, dropoff_lon])

    #Compute distances along lat, lon dimensions

    dlat = dropoff_lat - pickup_lat

    dlon = dropoff_lon - pickup_lon

    

    #Compute haversine distance

    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2

    

    return 2 * R_earth * np.arcsin(np.sqrt(a))



def add_airport_dist(dataset):

    """

    Return minumum distance from pickup or dropoff coordinates to each airport.

    JFK: John F. Kennedy International Airport

    EWR: Newark Liberty International Airport

    LGA: LaGuardia Airport

    """

    jfk_coord = (40.639722, -73.778889)

    ewr_coord = (40.6925, -74.168611)

    lga_coord = (40.77725, -73.872611)

    

    pickup_lat = dataset['pickup_latitude']

    dropoff_lat = dataset['dropoff_latitude']

    pickup_lon = dataset['pickup_longitude']

    dropoff_lon = dataset['dropoff_longitude']

    

    pickup_jfk  = distance(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 

    dropoff_jfk = distance(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 

    pickup_ewr  = distance(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])

    dropoff_ewr = distance(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 

    pickup_lga  = distance(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 

    dropoff_lga = distance(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 

    

    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)

    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)

    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    

    return dataset

    

def add_datetime_info(dataset):

    #Convert to datetime format

    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")

    

    dataset['hour'] = dataset.pickup_datetime.dt.hour

    dataset['day'] = dataset.pickup_datetime.dt.day

    dataset['month'] = dataset.pickup_datetime.dt.month

    dataset['weekday'] = dataset.pickup_datetime.dt.weekday

    dataset['year'] = dataset.pickup_datetime.dt.year

    

    return dataset



train = add_datetime_info(train)

train = add_airport_dist(train)

train['distance'] = distance(train['pickup_latitude'], train['pickup_longitude'], 

                                   train['dropoff_latitude'] , train['dropoff_longitude'])



train.head()
train.shape
train.sort_values(by = 'distance',ascending =False).head(100)
train.distance[(train.distance==0)].count()
train[(train.pickup_latitude != train.dropoff_latitude) &

              (train.pickup_longitude != train.dropoff_latitude) &

              (train.distance == 0)].count()
train[(train['distance']==0)&(train['fare_amount']==0)]
train = train.drop(train[(train['distance']==0)&(train['fare_amount']==0)].index, axis = 0)
train[(train['distance']==0)&(train['fare_amount']==0)]
# good
sns.distplot(a=train.fare_amount)
train['fare_amount'].skew()
# lets create a copy of train set

train_data = train.copy()
train_data.shape
train_data.drop(columns=['key', 'pickup_datetime'],inplace=True)
train_data.head()
train_data.sort_values(by = 'fare_amount',ascending =False).head(100)
corr_matrix = train_data.corr()
corr_matrix['fare_amount']
train_data.sort_values(by = 'passenger_count',ascending =True).head(10)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

plt.hist(train_data['passenger_count'],bins=15)

plt.xlabel('No of Passanger')

plt.ylabel('Frequency')
plt.figure(figsize=(10,7))

plt.scatter(x= train_data['passenger_count'],y = train_data['fare_amount'],s = 1.5)

plt.xlabel('No of Passengers')

plt.ylabel('fare_amount')
# lets drop it where there are fares for 0 passanger

print('Train_data befor ',train_data.shape)
train_data[(train_data['passenger_count']==0)&(train_data['fare_amount']>0)].sort_values(by = 'fare_amount',ascending=False)
train_data = train_data.drop(train_data[(train_data['passenger_count']==0)&(train_data['fare_amount']>0)].index, axis = 0)
train_data.shape
plt.figure(figsize=(15,7))

plt.scatter(x=train_data['day'], y=train_data['fare_amount'], s=1.5)

plt.xlabel('Date')

plt.ylabel('Fare')
plt.figure(figsize=(10,7))

plt.hist(train_data['hour'],bins=50)

plt.xlabel('Hour')

plt.ylabel('Frequency')
plt.figure(figsize=(15,7))

plt.scatter(x=train_data['distance'], y=train_data['fare_amount'], s=1.5)

plt.xlabel('Distance')

plt.ylabel('Fare')
train_data.head()
train_data.shape
train_data_org = train_data.copy()
del train_data
y = train_data_org['fare_amount']

train_data = train_data_org.drop(columns=['fare_amount'])
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler
data_pipeline = Pipeline([('rob_scale',RobustScaler())])
traindata_scaled = data_pipeline.fit_transform(train_data)
traindata_scaled
train_data.head()
traindata_scaled = pd.DataFrame(traindata_scaled,columns=train_data.columns,index=train_data.index)
traindata_scaled.head()
traindata_scaled.shape
test = add_datetime_info(test)

test = add_airport_dist(test)

test['distance'] = distance(test['pickup_latitude'], test['pickup_longitude'], 

                                   test['dropoff_latitude'] , test['dropoff_longitude'])



test.head()
test.drop(columns=['key', 'pickup_datetime'],inplace=True)
test.head()
test.shape
testdata_scaled = data_pipeline.fit_transform(test)
testdata_scaled
testdata_scaled = pd.DataFrame(testdata_scaled,columns=test.columns,index=test.index)
testdata_scaled.head()
print(train_data.shape,traindata_scaled.shape)
print(test.shape,testdata_scaled.shape)
traindata_scaled.to_csv('trained_scaled.csv',index=False)
train_data.to_csv('trained_data.csv',index=False)
testdata_scaled.to_csv('testdata_scaled.csv',index=False)

test.to_csv('test.csv',index=False)
y.to_csv('train_labels.csv',index=False)
train_data_org.to_csv('train_data_org.csv',index= False)
train_data_org.head()
train_data.head()
from sklearn.model_selection import train_test_split

import lightgbm as lgbm
x_train,x_test,y_train,y_test = train_test_split(traindata_scaled,y,random_state=123,test_size=0.10)
params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'nthread': 4,

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 5000 ,

        'bagging_freq': 20,

        'colsample_bytree': 0.6,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'zero_as_missing': True,

        'seed':0,

        'num_rounds':50000

    }



testdata_scaled.head()


# train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month','day','weekday'])

# valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month','day','weekday'])

# model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)



# prediction = model.predict(testdata_scaled, num_iteration = model.best_iteration)      







# submission = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv")

# submission['fare_amount'] = prediction

# submission.to_csv('lgbm_taxi_fare1.csv', index=False)











#submission.head()



#submission.head(20)
train_data.head()