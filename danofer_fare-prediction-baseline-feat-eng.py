# Data processing
import numpy as np
import pandas as pd
import datetime as dt
import random
# Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb

# check if date is a holiday:
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# Read data
train = pd.read_csv('../input/train.csv', nrows = 11123456)

#reading in sampled data misses the header row - need to debug
# n = 55000000 #Approx number of records in file as given in competition description
# s = 3123456 #desired sample size ~ a few million for now
# skip = sorted(random.sample(range(n+1),n-s))
# train = pd.read_csv('../input/train.csv', skiprows=skip,header=0) # sample data. Explicitly note that there's a header, otherwise we can skip it when sampling! 

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime,infer_datetime_format=True)

train.dropna(how = 'any', axis = 'rows',inplace=True) # drop some outliers/bad data\\
# todo: drop outliers from train based on distances? 
print("train shape:",train.shape)

train.head()
test = pd.read_csv('../input/test.csv')
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime,infer_datetime_format=True)
print("test size:", test.shape)
print('Test NaN values ?:')
print(test.isnull().sum()) # no nans in the test, unlike the train data

test.describe()
test.describe()
# Remove some potential location outliers:
## Taken from : https://www.kaggle.com/judesen/fare-prediction
## Ideally, check if test also has the extreme values or data errors.. 

# Latitude and longitude varies from -3116.28 to 2522.27 whereas the mean is around 40 (pickup_latitude, but goes for all the coordinates)
# This is probably due to a typo when data was gathered. Let's select a more reasonable value (2 times the standard deviation)
#columns_to_select = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
#for column in columns_to_select:
#    train = train.loc[(train[column] > train[column].mean() - train[column].std() * 2) & (train[column] < train[column].mean() + train[column].std() * 2)]

# Manually picking reasonable levels until I find a smarter way
train = train.loc[(train['fare_amount'] >= 0) & (train['fare_amount'] < 250)]
train = train.loc[(train['pickup_longitude'] > -90) & (train['pickup_longitude'] < 80) & (train['pickup_latitude'] > -80) & (train['pickup_latitude'] < 80)]
train = train.loc[(train['dropoff_longitude'] > -90) & (train['dropoff_longitude'] < 80) & (train['dropoff_latitude'] > -80) & (train['dropoff_latitude'] < 80)]


# Let's assume tax's can be mini-busses as well
train = train.loc[train['passenger_count'] <= 7]
train.describe()
combine = [train, test]
# test.dtypes
for dataset in combine:
    # Distance is expected to have an impact on the fare
    dataset['longitude_distance'] = abs(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    dataset['latitude_distance'] = abs(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    
    # Straight distance
    dataset['distance_travelled'] = (dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5
#     dataset['distance_travelled_sin'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
#     dataset['distance_travelled_cos'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    
#     dataset['distance_travelled_sin_sqrd'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
#     dataset['distance_travelled_cos_sqrd'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    
    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # c = 2 ⋅ atan2( √a, √(1−a) )
    # d = R ⋅ c
    R = 6371e3 # Metres
    phi1 = np.radians(dataset['pickup_latitude'])
    phi2 = np.radians(dataset['dropoff_latitude'])
    phi_chg = np.radians(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    delta_chg = np.radians(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    dataset['haversine'] = d
    
    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset['bearing'] = np.degrees(np.arctan2(y, x))
    
#     # Rhumb lines
    psi_chg = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = phi_chg / psi_chg
    d = (phi_chg + q ** 2 * delta_chg ** 2) ** .5 * R
    dataset['rhumb_lines'] = d
    
    # Maybe time of day matters? Obviously duration is a factor, but there is no data for time arrival
    # Features: hour of day (night vs day), month (some months may be in higher demand) 
    
#     dataset['pickup_datetime'] = pd.to_datetime(test['pickup_datetime']) # orig, used only test??
#     dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],infer_datetime_format=True) # new - may be wrong? 
    
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
#     dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset["dayofweek"] = dataset.pickup_datetime.dt.dayofweek
#     dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear

#     dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
    
    # check if holiday. Can try other calendars: https://stackoverflow.com/questions/29688899/pandas-checking-if-a-date-is-a-holiday-and-assigning-boolean-value
    cal = calendar()
    holidays = cal.holidays()
    dataset["usFedHoliday"] =  dataset.pickup_datetime.dt.date.astype('datetime64').isin(holidays)
    
    print(dataset.shape)
    
train.head(3)
train['distance_travelled'].describe()
test['distance_travelled'].describe()
train = train.loc[train['distance_travelled']< 0.6]
print(train.shape)
print(test.shape)
print(test.dropna().shape)
train.isna().sum()
test.isna().sum()
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# Drop irrelevant features
# We drop the key

# train_features_to_keep = ['haversine', 'fare_amount']
# train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)

train.drop(["key","pickup_datetime"],axis=1,inplace=True)
train.dropna(inplace=True)

# test_features_to_keep = ['key', 'haversine']
# test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)

test.drop(["pickup_datetime"],axis=1,inplace=True) # keep key in data for submission
# Let's prepare the test set
x_pred = test.drop('key', axis=1)

# Let's run XGBoost and predict those fares!
x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=126,test_size=0.16)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=350, 
                    early_stopping_rounds=15,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(3)
})

submission.to_csv('sub_fare.csv',index=False)
print(submission.shape)
submission.head()