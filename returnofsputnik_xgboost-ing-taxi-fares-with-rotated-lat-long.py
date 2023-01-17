import numpy as np 
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))
train_df =  pd.read_csv('../input/train.csv', nrows = 1_000_000)
train_df.dtypes
#Identify null values
print(train_df.isnull().sum())
#Drop rows with null values
train_df = train_df.dropna(how = 'any', axis = 'rows')
#Look at the first rows
train_df.head()
#Plot variables using only 1000 rows for efficiency
train_df.iloc[:1000].plot.scatter('pickup_longitude', 'pickup_latitude')
train_df.iloc[:1000].plot.scatter('dropoff_longitude', 'dropoff_latitude')

#Get distribution of values
train_df.describe()
#Clean dataset
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &
            (df.passenger_count > 0) & (df.passenger_count < 10)]

train_df = clean_df(train_df)
print(len(train_df))
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
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

def add_datetime_info(dataset):
    #Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    
    return dataset

train_df['distance'] = sphere_dist(train_df['pickup_latitude'], train_df['pickup_longitude'], 
                                   train_df['dropoff_latitude'] , train_df['dropoff_longitude'])

train_df = add_datetime_info(train_df)

train_df.head()
train_df.drop(columns=['key', 'pickup_datetime'], inplace=True)
train_df.head()
#y' = y*cos(a) - x*sin(a)
#x' = y*sin(a) + x*cos(a)
train_df['pickup_long_15'] = train_df['pickup_longitude']*np.cos(15* np.pi / 180) - train_df['pickup_latitude']*np.sin(15* np.pi/180)
train_df['pickup_long_30'] = train_df['pickup_longitude']*np.cos(30* np.pi / 180) - train_df['pickup_latitude']*np.sin(30* np.pi/180)
train_df['pickup_long_45'] = train_df['pickup_longitude']*np.cos(45* np.pi / 180) - train_df['pickup_latitude']*np.sin(45* np.pi/180)
train_df['pickup_long_60'] = train_df['pickup_longitude']*np.cos(60* np.pi / 180) - train_df['pickup_latitude']*np.sin(60* np.pi/180)
train_df['pickup_long_75'] = train_df['pickup_longitude']*np.cos(75* np.pi / 180) - train_df['pickup_latitude']*np.sin(75* np.pi/180)

train_df['pickup_lat_15'] = train_df['pickup_longitude']*np.sin(15* np.pi / 180) + train_df['pickup_latitude']*np.cos(15* np.pi/180)
train_df['pickup_lat_30'] = train_df['pickup_longitude']*np.sin(30* np.pi / 180) + train_df['pickup_latitude']*np.cos(30* np.pi/180)
train_df['pickup_lat_45'] = train_df['pickup_longitude']*np.sin(45* np.pi / 180) + train_df['pickup_latitude']*np.cos(45* np.pi/180)
train_df['pickup_lat_60'] = train_df['pickup_longitude']*np.sin(60* np.pi / 180) + train_df['pickup_latitude']*np.cos(60* np.pi/180)
train_df['pickup_lat_75'] = train_df['pickup_longitude']*np.sin(75* np.pi / 180) + train_df['pickup_latitude']*np.cos(75* np.pi/180)

train_df['dropoff_long_15'] = train_df['dropoff_longitude']*np.cos(15* np.pi / 180) - train_df['dropoff_latitude']*np.sin(15* np.pi/180)
train_df['dropoff_long_30'] = train_df['dropoff_longitude']*np.cos(30* np.pi / 180) - train_df['dropoff_latitude']*np.sin(30* np.pi/180)
train_df['dropoff_long_45'] = train_df['dropoff_longitude']*np.cos(45* np.pi / 180) - train_df['dropoff_latitude']*np.sin(45* np.pi/180)
train_df['dropoff_long_60'] = train_df['dropoff_longitude']*np.cos(60* np.pi / 180) - train_df['dropoff_latitude']*np.sin(60* np.pi/180)
train_df['dropoff_long_75'] = train_df['dropoff_longitude']*np.cos(75* np.pi / 180) - train_df['dropoff_latitude']*np.sin(75* np.pi/180)

train_df['dropoff_lat_15'] = train_df['dropoff_longitude']*np.sin(15* np.pi / 180) + train_df['dropoff_latitude']*np.cos(15* np.pi/180)
train_df['dropoff_lat_30'] = train_df['dropoff_longitude']*np.sin(30* np.pi / 180) + train_df['dropoff_latitude']*np.cos(30* np.pi/180)
train_df['dropoff_lat_45'] = train_df['dropoff_longitude']*np.sin(45* np.pi / 180) + train_df['dropoff_latitude']*np.cos(45* np.pi/180)
train_df['dropoff_lat_60'] = train_df['dropoff_longitude']*np.sin(60* np.pi / 180) + train_df['dropoff_latitude']*np.cos(60* np.pi/180)
train_df['dropoff_lat_75'] = train_df['dropoff_longitude']*np.sin(75* np.pi / 180) + train_df['dropoff_latitude']*np.cos(75* np.pi/180)
y = train_df['fare_amount']
train = train_df.drop(columns=['fare_amount'])

x_train,x_test,y_train,y_test = train_test_split(train,y,random_state=0,test_size=0.2)
def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                    dtrain=matrix_train,num_boost_round=100, 
                    early_stopping_rounds=100,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test)
#Read and preprocess test set
test_df =  pd.read_csv('../input/test.csv')
test_df['distance'] = sphere_dist(test_df['pickup_latitude'], test_df['pickup_longitude'], 
                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])
test_df = add_datetime_info(test_df)
#y' = y*cos(a) - x*sin(a)
#x' = y*sin(a) + x*cos(a)
test_df['pickup_long_15'] = test_df['pickup_longitude']*np.cos(15* np.pi / 180) - test_df['pickup_latitude']*np.sin(15* np.pi/180)
test_df['pickup_long_30'] = test_df['pickup_longitude']*np.cos(30* np.pi / 180) - test_df['pickup_latitude']*np.sin(30* np.pi/180)
test_df['pickup_long_45'] = test_df['pickup_longitude']*np.cos(45* np.pi / 180) - test_df['pickup_latitude']*np.sin(45* np.pi/180)
test_df['pickup_long_60'] = test_df['pickup_longitude']*np.cos(60* np.pi / 180) - test_df['pickup_latitude']*np.sin(60* np.pi/180)
test_df['pickup_long_75'] = test_df['pickup_longitude']*np.cos(75* np.pi / 180) - test_df['pickup_latitude']*np.sin(75* np.pi/180)

test_df['pickup_lat_15'] = test_df['pickup_longitude']*np.sin(15* np.pi / 180) + test_df['pickup_latitude']*np.cos(15* np.pi/180)
test_df['pickup_lat_30'] = test_df['pickup_longitude']*np.sin(30* np.pi / 180) + test_df['pickup_latitude']*np.cos(30* np.pi/180)
test_df['pickup_lat_45'] = test_df['pickup_longitude']*np.sin(45* np.pi / 180) + test_df['pickup_latitude']*np.cos(45* np.pi/180)
test_df['pickup_lat_60'] = test_df['pickup_longitude']*np.sin(60* np.pi / 180) + test_df['pickup_latitude']*np.cos(60* np.pi/180)
test_df['pickup_lat_75'] = test_df['pickup_longitude']*np.sin(75* np.pi / 180) + test_df['pickup_latitude']*np.cos(75* np.pi/180)

test_df['dropoff_long_15'] = test_df['dropoff_longitude']*np.cos(15* np.pi / 180) - test_df['dropoff_latitude']*np.sin(15* np.pi/180)
test_df['dropoff_long_30'] = test_df['dropoff_longitude']*np.cos(30* np.pi / 180) - test_df['dropoff_latitude']*np.sin(30* np.pi/180)
test_df['dropoff_long_45'] = test_df['dropoff_longitude']*np.cos(45* np.pi / 180) - test_df['dropoff_latitude']*np.sin(45* np.pi/180)
test_df['dropoff_long_60'] = test_df['dropoff_longitude']*np.cos(60* np.pi / 180) - test_df['dropoff_latitude']*np.sin(60* np.pi/180)
test_df['dropoff_long_75'] = test_df['dropoff_longitude']*np.cos(75* np.pi / 180) - test_df['dropoff_latitude']*np.sin(75* np.pi/180)

test_df['dropoff_lat_15'] = test_df['dropoff_longitude']*np.sin(15* np.pi / 180) + test_df['dropoff_latitude']*np.cos(15* np.pi/180)
test_df['dropoff_lat_30'] = test_df['dropoff_longitude']*np.sin(30* np.pi / 180) + test_df['dropoff_latitude']*np.cos(30* np.pi/180)
test_df['dropoff_lat_45'] = test_df['dropoff_longitude']*np.sin(45* np.pi / 180) + test_df['dropoff_latitude']*np.cos(45* np.pi/180)
test_df['dropoff_lat_60'] = test_df['dropoff_longitude']*np.sin(60* np.pi / 180) + test_df['dropoff_latitude']*np.cos(60* np.pi/180)
test_df['dropoff_lat_75'] = test_df['dropoff_longitude']*np.sin(75* np.pi / 180) + test_df['dropoff_latitude']*np.cos(75* np.pi/180)

test_key = test_df['key']
x_pred = test_df.drop(columns=['key', 'pickup_datetime'])

#Predict from test set
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
#Create submission file
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission