import pandas as pd

import numpy as np

from zipfile import ZipFile

import os

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from geopy import distance

import math



from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import Adam, SGD

from keras.callbacks import ModelCheckpoint
#base_path =  "./dataset/"

base_path = "../input/nyc-taxi-trip-duration/"
os.listdir(base_path)
data_df = pd.read_csv(base_path+"train.csv")
data_df.head()
data_df.describe()
sns.boxplot(x='vendor_id', y='trip_duration', data=data_df)
data_df[(data_df['trip_duration']>500000)].count()
data_df.drop(data_df[data_df['trip_duration'] > 500000].index, inplace=True)
sns.distplot(data_df['trip_duration'], bins=20000)
sns.distplot(np.log(data_df['trip_duration']), bins=20000)
data_df['trip_duration'] = data_df['trip_duration'].apply(math.log)
data_df = data_df[(data_df['trip_duration']<100000)]
data_df.reset_index(drop = True, inplace=True)
sns.countplot(x = 'vendor_id', data=data_df)
sns.boxplot(x='vendor_id', y='trip_duration', data=data_df)
sns.countplot(x = 'passenger_count', data=data_df)
sns.boxplot(x = 'passenger_count', y = 'trip_duration', data=data_df)
sns.countplot(x = 'store_and_fwd_flag', data=data_df)
data_df[data_df['store_and_fwd_flag']=='Y']['trip_duration'].describe()
sns.distplot(data_df[data_df['store_and_fwd_flag']=='Y']['trip_duration'], bins=1000)
data_df[data_df['store_and_fwd_flag']=='N']['trip_duration'].describe()
sns.distplot(data_df[data_df['store_and_fwd_flag']=='N']['trip_duration'], bins=1000)
sns.boxplot(x = 'store_and_fwd_flag', y = 'trip_duration', data=data_df)
def strtodatetime(x):

    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
data_df['pickup_datetime'] = data_df['pickup_datetime'].apply(strtodatetime)
def dayofweek(x):

    return x.weekday()
data_df['day_of_week'] = data_df['pickup_datetime'].apply(dayofweek)
sns.boxenplot(x='day_of_week', y='trip_duration', data=data_df)
distance.distance((data_df['pickup_latitude'].iloc[0],

                  data_df['pickup_longitude'].iloc[0]),

                 (data_df['dropoff_latitude'].iloc[0],

                 data_df['dropoff_longitude'].iloc[0])).m
def dist(x):

   return distance.distance((x[0], x[1]),

                 (x[2], x[3])).m
data_df['dist'] = data_df[['pickup_latitude',

         'pickup_longitude',

         'dropoff_latitude',

         'dropoff_longitude']].apply(lambda x:dist(x), axis=1)
sns.distplot(data_df['dist'], bins=1000)
data_df[(data_df['pickup_latitude']==data_df['dropoff_latitude']) &

        (data_df['pickup_longitude']==data_df['dropoff_longitude'])].count()
#data_df.drop(data_df[(data_df['pickup_latitude']==data_df['dropoff_latitude']) &

#        (data_df['pickup_longitude']==data_df['dropoff_longitude'])].index,

#            inplace=True)
data_df['dist'].replace(to_replace=0, value=1, inplace=True)
#log

data_df['dist'] = data_df['dist'].apply(math.log)
sns.distplot(data_df['dist'], bins=1000)
#currently handling data only day wise

MAX_SECONDS_IN_DAY = 24*60*60

def timetosectosincosday(x):

    #x = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    initial_date = datetime(x.year, x.month, x.day, 0 , 0, 0) # 1 is for day

    sec = (x-initial_date).total_seconds()

    sin = math.sin(2*math.pi*(sec/MAX_SECONDS_IN_DAY))

    cos = math.cos(2*math.pi*(sec/MAX_SECONDS_IN_DAY))

    return sin, cos
data_df['pickup_sin_sec'] ,data_df['pickup_cos_sec'] = zip(*data_df['pickup_datetime'].map(timetosectosincosday))
#data day of week

MAX_DAY_IN_WEEK = 6 # from 0-6

def dayofweektosincosday(day):

    #x = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    sin = math.sin(2*math.pi*(day/MAX_DAY_IN_WEEK))

    cos = math.cos(2*math.pi*(day/MAX_DAY_IN_WEEK))

    return sin, cos
data_df['sin_day'] ,data_df['cos_day'] = zip(*data_df['day_of_week'].map(dayofweektosincosday))
df_processed =  data_df[['vendor_id',

                        'passenger_count',

                        'pickup_sin_sec',

                        'pickup_cos_sec',

                        'pickup_longitude',

                        'pickup_latitude',

                        'dropoff_longitude',

                        'dropoff_latitude',

                        'sin_day',

                        'cos_day',

                        'dist',

                        'trip_duration']]
X = data_df[['vendor_id',

            'passenger_count',

            'pickup_sin_sec',

            'pickup_cos_sec',

            'pickup_longitude',

            'pickup_latitude',

            'dropoff_longitude',

            'dropoff_latitude',

            'sin_day',

            'cos_day',

            'dist']]
y = data_df[['trip_duration']]
standardScalarX = StandardScaler().fit(X)

X  = standardScalarX.transform(X)
standardScalarY = StandardScaler().fit(y)

y = standardScalarY.transform(y)
X.shape, y.shape
model = Sequential()

model.add(Dense(128, activation = 'relu',input_shape=(X.shape[1],)))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.00001),

              metrics=['mean_squared_error'], 

              loss='mean_squared_error')
#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
history = model.fit(X, y,

                   batch_size=32,

                   validation_split=0.2,

                   epochs=50, 

                   callbacks=callbacks_list)
history.history.keys()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(["train", "validation"])

plt.savefig('loss.png')
model.save('model.hdf5')
os.listdir(base_path)
test_df = pd.read_csv(base_path+"test.csv")
test_df.head()
test_df['pickup_datetime'] = test_df['pickup_datetime'].apply(strtodatetime)
test_df['day_of_week'] = test_df['pickup_datetime'].apply(dayofweek)
test_df['sin_day'] ,test_df['cos_day'] = zip(*test_df['day_of_week'].map(dayofweektosincosday))
test_df['dist'] = test_df[['pickup_latitude',

         'pickup_longitude',

         'dropoff_latitude',

         'dropoff_longitude']].apply(lambda x:dist(x), axis=1)
test_df['dist'].replace(to_replace=0, value=1, inplace=True)
test_df['dist'] = test_df['dist'].apply(math.log)
test_df['pickup_sin_sec'] ,test_df['pickup_cos_sec'] = zip(*test_df['pickup_datetime'].map(timetosectosincosday))
test_df_processed =  test_df[['vendor_id',

                            'passenger_count',

                            'pickup_sin_sec',

                            'pickup_cos_sec',

                            'pickup_longitude',

                            'pickup_latitude',

                            'dropoff_longitude',

                            'dropoff_latitude',

                            'sin_day',

                            'cos_day',

                            'dist']]
test_data =  standardScalarX.transform(test_df_processed)
y_pred = model.predict(test_data)
y_pred =standardScalarY.inverse_transform(y_pred)

y_pred = np.exp(y_pred)
result = test_df[['id']]
result['trip_duration'] = pd.DataFrame(data=y_pred, columns=['trip_duration'])
result.head()
result.to_csv("submission.csv", index = False)