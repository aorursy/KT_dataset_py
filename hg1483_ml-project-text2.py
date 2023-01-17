import numpy as np

import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import EarlyStopping

from keras import optimizers

from keras import regularizers
TRAIN_PATH = '../input/train.csv'

TEST_PATH = '../input/test.csv'

SUBMISSION_NAME = 'submission.csv'

DATASET_SIZE = 200000

# datatypes = {'key': 'str', 

#               'fare_amount': 'float32',

#               'pickup_datetime': 'str', 

#               'pickup_longitude': 'float32',

#               'pickup_latitude': 'float32',

#               'dropoff_longitude': 'float32',

#               'dropoff_latitude': 'float32',

#               'passenger_count': 'uint8'}

train = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE)

test = pd.read_csv(TEST_PATH)

# train.head()

# test=test.drop('key', axis=1)

train.head()

print('Old size: %d' % len(train))

train = train[train.fare_amount>=0]

print('New size: %d' % len(train))

print('Old size: %d' % len(train))

train = train.dropna(how = 'any', axis = 'rows')

print('New size: %d' % len(train))
# def select_within_boundingbox(df, BB):

#     return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \

#            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \

#            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \

#            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

            

# # load image of NYC map

# BB = (-74.5, -72.8, 40.5, 41.8)

# print('Old size: %d' % len(train))

# train = train[select_within_boundingbox(train, BB)]

# print('New size: %d' % len(train))
print('Old size: %d' % len(train))

def clean(df):

    # Delimiter lats and lons to NY only

    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]

    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]

    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]

    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]

    # Remove possible outliers

    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 250)]

    # Remove inconsistent values

    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]

    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]

    

    return df

train = clean(train)

print('New size: %d' % len(train))

train.head()
def late_night (row):

    if (row['hour'] <= 6) or (row['hour'] >20):

        return 1

    else:

        return 0

def night (row):

    if ((row['hour'] <= 20) and (row['hour'] > 16)) and (row['weekday'] < 5):

        return 1

    else:

        return 0    

def day_time(row):

    if (row['hour'] <= 16) and (row['hour'] >6):

        return 1

    else:

        return 0

def spring(row):

    if (row['month'] >=4) and (row['month'] <7):

        return 1

    else:

        return 0  

def summer(row):

    if (row['month'] >=7) and (row['month'] <10):

        return 1

    else:

        return 0

def fall(row):

    if (row['month'] >=10) and (row['month'] <=12):

        return 1

    else:

        return 0  

def winter(row):

    if (row['month'] >=1) and (row['month'] <4):

        return 1

    else:

        return 0  



def add_time_features(df):

    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')

    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)

    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)

    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)

    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)

    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())

    df['pickup_datetime'] =  df['pickup_datetime'].apply(lambda x: str(x))

    df['night'] = df.apply (lambda x: night(x), axis=1)

    df['late_night'] = df.apply (lambda x: late_night(x), axis=1)

    df['day_time'] = df.apply (lambda x: day_time(x), axis=1)

    df['spring']=df.apply (lambda x: spring(x), axis=1)

    df['summer']=df.apply (lambda x: summer(x), axis=1)

    df['fall']=df.apply (lambda x: fall(x), axis=1)

    df['winter']=df.apply (lambda x: winter(x), axis=1)

    # Drop 'pickup_datetime' as we won't need it anymore

    df = df.drop('pickup_datetime', axis=1)

    return df

train = add_time_features(train)

test = add_time_features(test)

train.head(100)
# print(train.year.min(),train.year.max())

# print(test.year.min(),test.year.max())

def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295 # Pi/180

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
def add_coordinate_features(df):

    lat1 = df['pickup_latitude']

    lat2 = df['dropoff_latitude']

    lon1 = df['pickup_longitude']

    lon2 = df['dropoff_longitude']

    # Add new features

    df['latdiff'] = (lat1 - lat2)

    df['londiff'] = (lon1 - lon2)

    return df
def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):

    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)

def add_distances_features(df):

    # Add distances from airpot and downtown

    ny = (-74.0063889, 40.7141667)

    jfk = (-73.7822222222, 40.6441666667)

    ewr = (-74.175, 40.69)

    lgr = (-73.87, 40.77)

    

    lat1 = df['pickup_latitude']

    lat2 = df['dropoff_latitude']

    lon1 = df['pickup_longitude']

    lon2 = df['dropoff_longitude']

    

    df['euclidean'] = (df['latdiff'] ** 2 + df['londiff'] ** 2) ** 0.5

    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)

    

    df['downtown_pickup_distance'] = manhattan(ny[1], ny[0], lat1, lon1)

    df['downtown_dropoff_distance'] = manhattan(ny[1], ny[0], lat2, lon2)

    df['jfk_pickup_distance'] = manhattan(jfk[1], jfk[0], lat1, lon1)

    df['jfk_dropoff_distance'] = manhattan(jfk[1], jfk[0], lat2, lon2)

    df['ewr_pickup_distance'] = manhattan(ewr[1], ewr[0], lat1, lon1)

    df['ewr_dropoff_distance'] = manhattan(ewr[1], ewr[0], lat2, lon2)

    df['lgr_pickup_distance'] = manhattan(lgr[1], lgr[0], lat1, lon1)

    df['lgr_dropoff_distance'] = manhattan(lgr[1], lgr[0], lat2, lon2)

    return df


add_coordinate_features(train)

add_coordinate_features(test)

train = add_distances_features(train)

test = add_distances_features(test)

train.head()
dropped_columns = ['pickup_longitude', 'pickup_latitude', 

                   'dropoff_longitude', 'dropoff_latitude']

train_clean = train.drop(dropped_columns, axis=1)

test_clean = test.drop(dropped_columns + ['key', 'passenger_count'], axis=1)



# peek data

train_clean.head()
train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

# Get labels

train_labels = train_df['fare_amount'].values

validation_labels = validation_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)

validation_df = validation_df.drop(['fare_amount'], axis=1)
scaler = preprocessing.MinMaxScaler()

train_df_scaled = scaler.fit_transform(train_df)

validation_df_scaled = scaler.transform(validation_df)

test_scaled = scaler.transform(test_clean)
BATCH_SIZE = 256

EPOCHS = 20

LEARNING_RATE = 0.001

model = Sequential()

model.add(Dense(256, activation='relu', input_dim=train_df_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(8, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1))

adam = optimizers.Adam(lr=LEARNING_RATE)

model.compile(loss='mse', optimizer=adam, metrics=['mae'])
history = model.fit(x=train_df_scaled, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, 

                    verbose=1, validation_data=(validation_df_scaled, validation_labels), 

                    shuffle=True)
plt.figure(figsize=(20,10))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):

    df = pd.DataFrame(prediction, columns=[prediction_column])

    df[id_column] = raw_test[id_column]

    df[[id_column, prediction_column]].to_csv((file_name), index=False)

    print('Output complete')

output_submission(test, prediction, 'key', 'fare_amount', SUBMISSION_NAME)