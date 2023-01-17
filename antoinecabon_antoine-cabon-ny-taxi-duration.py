# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler



from threading import Thread



%matplotlib inline

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
df.info()
df.groupby('vendor_id').count()['id'].plot.bar()
one = df.groupby('vendor_id').count()['id'].iloc[0:1]

two = df.groupby('vendor_id').count()['id'].iloc[1:2]



print('Vendor_id 2 has received ' , np.subtract(two, one).values[0], 'more bookings then Vendor_id 1.')
f, axes = plt.subplots(2,figsize=(30, 15), sharex=False, sharey = False)

sns.distplot(df['pickup_latitude'], label = 'pickup_latitude',color="y",bins = 100, ax=axes[0], hist=False).tick_params(labelsize=20)

sns.distplot(df['pickup_longitude'], label = 'pickup_longitude',color="y",bins =100, ax=axes[1], hist=False).tick_params(labelsize=20)
f, axes = plt.subplots(2,figsize=(30, 15), sharex=False, sharey = False)

sns.distplot(df['dropoff_latitude'], label = 'dropoff_latitude',color="y",bins = 100, ax=axes[0], hist=False).tick_params(labelsize=20)

sns.distplot(df['dropoff_longitude'], label = 'dropoff_longitude',color="y",bins =100, ax=axes[1], hist=False).tick_params(labelsize=20)
df['passenger_count'].value_counts()[0:6].plot(kind='pie', subplots=True, figsize=(8, 8))



passenger_1_percentage = df['passenger_count'].value_counts().values[0]/df['passenger_count'].count()

print(passenger_1_percentage)
df.isna().sum()
# approximate radius of earth in km

R = 6371.0



#df

lat1 = np.radians(df['pickup_latitude'])

lon1 = np.radians(df['pickup_longitude'])

lat2 = np.radians(df['dropoff_latitude']) 

lon2 = np.radians(df['dropoff_longitude'])



dlon = lon2 - lon1

dlat = lat2 - lat1



a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))



distance = R * c



distance_df = pd.DataFrame({'travel_distance_km':distance})

df['travel_distance_km'] = distance_df



print(np.mean(distance_df))



#Test



lat1 = np.radians(test['pickup_latitude'])

lon1 = np.radians(test['pickup_longitude'])

lat2 = np.radians(test['dropoff_latitude']) 

lon2 = np.radians(test['dropoff_longitude'])



dlon = lon2 - lon1

dlat = lat2 - lat1



a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))



distance = R * c



distance_df = pd.DataFrame({'travel_distance_km':distance})

test['travel_distance_km'] = distance_df

#df

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])



df['hour_pickup'] = np.array(df['pickup_datetime'].dt.hour)

df['day_pickup'] = np.array(df['pickup_datetime'].dt.day_name())



#test

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])



test['hour_pickup'] = np.array(test['pickup_datetime'].dt.hour)

test['day_pickup'] = np.array(test['pickup_datetime'].dt.day_name())
sns.set(rc={'figure.figsize':(15,10)})

sns.distplot(df['travel_distance_km'],hist=False)
sns.distplot(df[ 

        (df['travel_distance_km'] >= 0) & 

        (df['travel_distance_km'] < 200)

                                        ]['travel_distance_km'],hist=False)
category_dict = {'0-5': 0,'5-10': 0, '10-15': 0, '15-20': 0, '20-25':0 }



for x in np.array(df[(df['travel_distance_km'] >= 0) & (df['travel_distance_km'] < 25)]['travel_distance_km']): 

    

    if x <= 5: 

        category_dict['0-5'] = category_dict['0-5'] + 1

        

    elif x > 5 and x <= 10 : 

        category_dict['5-10'] = category_dict['5-10'] + 1

    

    elif x > 10 and x <= 15 :

        category_dict['10-15'] = category_dict['10-15'] + 1



    elif x > 15 and x <= 20 :

        category_dict['15-20'] = category_dict['15-20'] + 1

    

    elif x > 20 and x <= 25 :

        category_dict['20-25'] = category_dict['20-25'] + 1

        

pd.DataFrame(category_dict, index=[0]).T.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title('Percentage of Trips Between 0 to 25 km')



print('% of trips between 0 to 5 km for the trip distance between 0 to 25 km dataset : ', category_dict['0-5']/sum(category_dict.values()))
df[(df['travel_distance_km'] >= 0) & (df['travel_distance_km'] < 25) & (df['trip_duration'] < 20000) ]['trip_duration'].hist(bins=300)



plt.title('Duration for Trips between 0 to 25km')

plt.xlabel('Duration in Seconds')

plt.ylabel('Frequency')



print('mode : ', df[(df['travel_distance_km'] >= 0) & (df['travel_distance_km'] < 25) & (df['trip_duration'] < 20000) ]['trip_duration'].mode()[0])

print('mean :' , df[(df['travel_distance_km'] >= 0) & (df['travel_distance_km'] < 25) & (df['trip_duration'] < 20000) ]['trip_duration'].mean())

print('median :' , df[(df['travel_distance_km'] >= 0) & (df['travel_distance_km'] < 25) & (df['trip_duration'] < 20000) ]['trip_duration'].median())



np.log(df[(df['travel_distance_km'] >= 0) & (df['travel_distance_km'] < 25)]['trip_duration']).plot.hist(bins=300)

plt.title('log(Duration for Trips between 0 to 25km)')

plt.xlabel('log(Duration in Seconds)')

plt.ylabel('Frequency')
sns.scatterplot(

    x=df[(df['travel_distance_km'] > 0) & (df['travel_distance_km'] < 25) & (df['trip_duration'] < 20000)]['trip_duration'], 

    y=df[(df['travel_distance_km'] > 0) & (df['travel_distance_km'] < 25) & (df['trip_duration'] < 20000)]['travel_distance_km'])
df['day_pickup'] = df['day_pickup'].astype('category').cat.codes

df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category').cat.codes



test['day_pickup'] = test['day_pickup'].astype('category').cat.codes

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].astype('category').cat.codes
scaler = StandardScaler()



scaler.fit(df[['vendor_id','passenger_count', 

        'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

        'travel_distance_km', 'hour_pickup',

       'day_pickup']])



scaled_features = scaler.transform(df[['vendor_id','passenger_count', 

        'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

        'travel_distance_km', 'hour_pickup',

       'day_pickup']])



df_feat = pd.DataFrame(scaled_features, columns=['vendor_id','passenger_count', 

        'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

        'travel_distance_km', 'hour_pickup',

       'day_pickup'])





scaler_test = StandardScaler()



scaler_test.fit(test[['vendor_id','passenger_count', 

        'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

        'travel_distance_km', 'hour_pickup',

       'day_pickup']])



scaled_features_test = scaler.transform(test[['vendor_id','passenger_count', 

        'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

        'travel_distance_km', 'hour_pickup',

       'day_pickup']])



df_feat_test = pd.DataFrame(scaled_features_test, columns=['vendor_id','passenger_count', 

        'pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',

        'travel_distance_km', 'hour_pickup',

       'day_pickup'])



test = pd.concat([test['id'],df_feat_test], axis=1)

X = df_feat



y = np.log(df['trip_duration'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)
X_train.shape
rmse_rfr = []



def func_rfr(num): 

    rfr = RandomForestRegressor(n_estimators=num)

    rfr.fit(X_train, y_train)

    pred = rfr.predict(X_valid)

    rmse_rfr.append(np.sqrt(mean_squared_error(y_valid, pred)))
if __name__ == '__main__':

    Thread(target = func_rfr(20)).start()

    Thread(target = func_rfr(50)).start()

    Thread(target = func_rfr(100)).start()
sns.lineplot(y=rmse_rfr, x=[20,50,100], markers=True)
rfr = RandomForestRegressor(n_estimators=100)

rfr.fit(X_train, y_train)

pred = rfr.predict(X_valid)

score_rfr = cross_val_score(rfr, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
print('cross_val_score average: ', abs(np.mean(score_rfr)))

print('MAE: ',mean_absolute_error(y_valid, pred))

print('MSE: ',mean_squared_error(y_valid, pred))

print('RMSE: ',np.sqrt(mean_squared_error(y_valid, pred)))
pred = rfr.predict(test[['vendor_id','passenger_count',

       'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

       'dropoff_latitude', 'store_and_fwd_flag', 'travel_distance_km',

       'hour_pickup', 'day_pickup']])
arr_id = test['id']

submission = pd.DataFrame({'id': arr_id, 'trip_duration': np.exp(pred)})

print(submission)



submission.to_csv("submit_file.csv", index=False)