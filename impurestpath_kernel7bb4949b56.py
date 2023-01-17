# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('seaborn-whitegrid')

df_train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000,parse_dates=["pickup_datetime"])

df_train.head()
df_train.describe()
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv",parse_dates=["pickup_datetime"])

df_test.head()
df_test.describe()
print('Old size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount>=0]

print('New size: %d' % len(df_train))
print('Old size: %d' % len(df_train))

df_train = df_train.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(df_train))
df_train[df_train.fare_amount < 80].fare_amount.hist(bins=100)

plt.xlabel('fare $USD')
df_train['diff_long'] = (df_train.dropoff_longitude - df_train.pickup_longitude).abs()

df_train['diff_long'].describe()
df_train['diff_lat'] = (df_train.dropoff_latitude - df_train.pickup_latitude).abs()

df_train['diff_lat'].describe()
print('Old size: %d' % len(df_train))

df_train = df_train[(df_train.diff_long < 5.0) & (df_train.diff_lat < 5.0)]

print('New size: %d' % len(df_train))
df_train['year'] = df_train.pickup_datetime.apply(lambda t: t.year)

df_train['month'] = df_train.pickup_datetime.apply(lambda t: t.month)

df_train['weekday'] = df_train.pickup_datetime.apply(lambda t: t.weekday())

df_train['hour'] = df_train.pickup_datetime.apply(lambda t: t.hour)
df_train.describe()
df_train[['fare_amount', 'hour']].groupby(['hour'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)
df_train[['fare_amount', 'weekday']].groupby(['weekday'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)
df_train[['fare_amount', 'year']].groupby(['year'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)
def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295 # Pi/180

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
df_train['distance'] = distance(df_train.pickup_latitude, df_train.pickup_longitude, \

                                      df_train.dropoff_latitude, df_train.dropoff_longitude)
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter('distance', 'fare_amount',alpha=0.1)
print('Old size: %d' % len(df_train))

df_train = df_train[(df_train.distance >= 0.1)]

print('New size: %d' % len(df_train))
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter('distance', 'fare_amount',alpha=0.1)
print('Old size: %d' % len(df_train))

df_train = df_train[(df_train.distance <= 50)]

print('New size: %d' % len(df_train))
plot = df_train.plot.scatter('distance', 'fare_amount',alpha=0.1)
print('Old size: %d' % len(df_train))

df_train = df_train[(df_train.fare_amount <= 200)]

print('New size: %d' % len(df_train))
plot = df_train.plot.scatter('distance', 'fare_amount',alpha=0.1)
def add_all_dist(dataset):

    """

    Return minumum distance from pickup or dropoff coordinates to each airport.

    JFK: John F. Kennedy International Airport

    EWR: Newark Liberty International Airport

    LGA: LaGuardia Airport

    """

    jfk_coord = (40.639722, -73.778889)

    ewr_coord = (40.6925, -74.168611)

    lga_coord = (40.77725, -73.872611)

    center_coord = (40.7141667, -74.0063889)

    

    pickup_lat = dataset['pickup_latitude']

    dropoff_lat = dataset['dropoff_latitude']

    pickup_lon = dataset['pickup_longitude']

    dropoff_lon = dataset['dropoff_longitude']

    

    pickup_jfk = distance(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 

    dropoff_jfk = distance(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 

    pickup_ewr = distance(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])

    dropoff_ewr = distance(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 

    pickup_lga = distance(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 

    dropoff_lga = distance(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 

    pickup_center = distance(center_coord[0], center_coord[1], dropoff_lat, dropoff_lon)

    dropoff_center = distance(pickup_lat, pickup_lon, center_coord[0], center_coord[1])

    

    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)

    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)

    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    dataset['ctr_pick_dist'] = pickup_center

    dataset['ctr_drop_dist'] = dropoff_center

    

    return dataset
df_train = add_all_dist(df_train)
df_train.describe()
plt.figure(figsize=(15,8))

sns.heatmap(df_train.drop(['key','pickup_datetime'],axis=1).corr(),annot=True,fmt='.4f')
features = ['year', 'hour','month', 'distance','passenger_count', 'jfk_dist','ewr_dist','lga_dist','ctr_pick_dist','ctr_drop_dist']

X = df_train[features].values

y = df_train['fare_amount'].values
df_test['year'] = df_test.pickup_datetime.apply(lambda t: t.year)

df_test['month'] = df_test.pickup_datetime.apply(lambda t: t.month)

df_test['hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)

df_test['distance'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \

                                      df_test.dropoff_latitude, df_test.dropoff_longitude)

df_test = add_all_dist(df_test)
X_kaggle_test = df_test[features].values
from sklearn.model_selection import train_test_split

import xgboost as xgb



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.3)



def XGBmodel(x_train,x_test,y_train,y_test):

    matrix_train = xgb.DMatrix(x_train,label=y_train)

    matrix_test = xgb.DMatrix(x_test,label=y_test)

    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},

                    dtrain=matrix_train,num_boost_round=200, 

                    early_stopping_rounds=100,evals=[(matrix_test,'test')])

    return model



model = XGBmodel(X_train,X_test,y_train,y_test)

prediction = model.predict(xgb.DMatrix(X_kaggle_test), ntree_limit = model.best_ntree_limit)



submission = pd.DataFrame({

        "key": df_test['key'],

        "fare_amount": prediction.round(2)

})



submission.to_csv('submission.csv',index=False)

submission
