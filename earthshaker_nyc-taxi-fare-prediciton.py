# Data Exploration

# Importing various libraries

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
import xgboost as xgb
from tqdm import tqdm
df_train = pd.read_csv("../input/train.csv", nrows=1000000)
def CleanData(df_train):
    #print(df_train.head())
    # Removing null entries

    df_train = df_train[df_train['dropoff_latitude'].isnull() == False]

    # Removing odd passenger counts

    df_train = df_train[df_train.passenger_count < 8]
    df_train = df_train[df_train.passenger_count >= 1]

    # Removing negative fares

    df_train = df_train[df_train.fare_amount > 0]

    # These cut off values were done by trial and error, and observing the bounding box of NYC on Google Maps

    df_train = df_train[df_train['pickup_longitude'] > -75]
    df_train = df_train[df_train['pickup_longitude'] < -73]
    df_train = df_train[df_train['pickup_latitude'] < 42]
    df_train = df_train[df_train['pickup_latitude'] > 40]


    df_train = df_train[df_train['dropoff_longitude'] > -75]
    df_train = df_train[df_train['dropoff_longitude'] < -73]
    df_train = df_train[df_train['dropoff_latitude'] < 42]
    df_train = df_train[df_train['dropoff_latitude'] > 40]

    return df_train
df_train = CleanData(df_train)
def HourGroup(hour):
    group = 0
    
    # 0:00 - 7:00
    if hour >= 0 and hour <=7:
        group = 1
    # 18:00 - 22:00
    elif hour >= 18 and hour <= 22:
        group = 2
    # other hours
    else:
        group = 3
        
    return group
def DistanceCal(df_train, test=1):
    # Got it from https://www.kaggle.com/pavanraj159/nyc-taxi-fare-time-series-forecasting
    # Calculating distnace between coordinates
    R = 6373.0

    pickup_lat  = np.radians(df_train["pickup_latitude"])
    pickup_lon  = np.radians(df_train["pickup_longitude"])
    dropoff_lat = np.radians(df_train["dropoff_latitude"])
    dropoff_lon = np.radians(df_train["dropoff_longitude"])

    dist_lon = dropoff_lon - pickup_lon
    dist_lat = dropoff_lat - pickup_lat

    #Formula
    a = (np.sin(dist_lat/2))**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * (np.sin(dist_lon/2))**2 
    c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) 
    d = R * c #(where R is the radius of the Earth)

    df_train['distance'] = d
    
    if test:
        df_train = df_train[df_train['distance'] > 1]
        df_train = df_train[df_train['distance'] < 110]
    
    return df_train
# Taken from https://www.kaggle.com/aiswaryaramachandran/eda-and-feature-engineering

nyc_airports={'JFK':{'min_lng':-73.8352,
     'min_lat':40.6195,
     'max_lng':-73.7401, 
     'max_lat':40.6659},
              
    'EWR':{'min_lng':-74.1925,
            'min_lat':40.6700, 
            'max_lng':-74.1531, 
            'max_lat':40.7081

        },
    'LaGuardia':{'min_lng':-73.8895, 
                  'min_lat':40.7664, 
                  'max_lng':-73.8550, 
                  'max_lat':40.7931
        
    }
    
}

def isAirport(latitude,longitude,airport_name='JFK'):
    
    if latitude>=nyc_airports[airport_name]['min_lat'] and latitude<=nyc_airports[airport_name]['max_lat'] and longitude>=nyc_airports[airport_name]['min_lng'] and longitude<=nyc_airports[airport_name]['max_lng']:
        return 1
    else:
        return 0
def FeatureCreation(df_train, test=1):

    df_train['date'] = df_train['pickup_datetime'].apply(lambda x : x[:-12])
    df_train['time'] = df_train['pickup_datetime'].apply(lambda x : x[11:-4])

    # Features to use
    
    df_train['month'] = df_train['date'].apply(lambda x : int(x.split("-")[1]))
    df_train['date_num'] = df_train['date'].apply(lambda x : int(x.split("-")[2]))
    df_train['hour'] = df_train['time'].apply(lambda x : int(x.split(":")[0]))

    df_train['hour_group'] = df_train['hour'].apply(lambda x : HourGroup(x))

    df_train['day'] = df_train['date'].apply(lambda x : datetime.datetime.strptime(x.strip(), "%Y-%m-%d").weekday())
    
    df_train['is_pickup_JFK']=df_train.apply(lambda x:isAirport(x['pickup_latitude'],x['pickup_longitude'],'JFK'),axis=1)
    df_train['is_dropoff_JFK']=df_train.apply(lambda x:isAirport(x['dropoff_latitude'],x['dropoff_longitude'],'JFK'),axis=1)

    df_train['is_pickup_EWR']=df_train.apply(lambda x:isAirport(x['pickup_latitude'],x['pickup_longitude'],'EWR'),axis=1)
    df_train['is_dropoff_EWR']=df_train.apply(lambda x:isAirport(x['dropoff_latitude'],x['dropoff_longitude'],'EWR'),axis=1)
    
    df_train['is_pickup_la_guardia']=df_train.apply(lambda x:isAirport(x['pickup_latitude'],x['pickup_longitude'],'LaGuardia'),axis=1)
    df_train['is_dropoff_la_guardia']=df_train.apply(lambda x:isAirport(x['dropoff_latitude'],x['dropoff_longitude'],'LaGuardia'),axis=1)
    
    df_train = DistanceCal(df_train, test)
    
    return df_train
df_train = FeatureCreation(df_train)
features = ['day', 'hour', 'distance', 'passenger_count', 'month', 'hour_group', \
            'date_num', 'is_pickup_JFK', 'is_dropoff_JFK', 'is_pickup_EWR', 'is_dropoff_EWR', \
           'is_pickup_la_guardia', 'is_dropoff_la_guardia']

x = df_train[features]
y = df_train['fare_amount']

x.shape,y.shape
# create training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# testing the model

#Cross-validation
params = {
    # Parameters that we are going to tune.
    'max_depth': 8, #Result of tuning with CV
    'eta':.03, #Result of tuning with CV
    'subsample': 1, #Result of tuning with CV
    'colsample_bytree': 0.8, #Result of tuning with CV
    # Other parameters
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 1
}

def XGBmodel(x_train,x_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(X_train,X_test,y_train,y_test,params)
df_test = pd.read_csv('../input/test.csv')
df_test = FeatureCreation(df_test,0)
print(max(df_test['pickup_longitude']))
print(max(df_test['pickup_longitude']))
print(max(df_test['pickup_latitude']))
print(max(df_test['pickup_latitude']))

print(max(df_test['dropoff_longitude']))
print(max(df_test['dropoff_longitude']))
print(max(df_test['dropoff_latitude']))
print(max(df_test['dropoff_latitude']))
features = ['day', 'hour', 'distance', 'passenger_count', 'month', 'hour_group', \
            'date_num', 'is_pickup_JFK', 'is_dropoff_JFK', 'is_pickup_EWR', 'is_dropoff_EWR', \
           'is_pickup_la_guardia', 'is_dropoff_la_guardia']
X = df_test[features].values

X_pred = pd.DataFrame(X, columns=['day', 'hour', 'distance', 'passenger_count', 'month', 'hour_group', \
            'date_num', 'is_pickup_JFK', 'is_dropoff_JFK', 'is_pickup_EWR', 'is_dropoff_EWR', \
           'is_pickup_la_guardia', 'is_dropoff_la_guardia'])
#Predict from test set
prediction = model.predict(xgb.DMatrix(X_pred), ntree_limit = model.best_ntree_limit)
df_pred = pd.DataFrame(prediction, columns=["fare_amount"])

df_key = df_test['key']

result = pd.concat([df_key,df_pred], axis=1, sort=False)
result.to_csv("my_submission.csv", index=False)
result.head()