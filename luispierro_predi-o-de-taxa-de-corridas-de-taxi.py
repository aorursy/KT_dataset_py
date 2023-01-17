# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras import regularizers, optimizers

from keras.layers.normalization import BatchNormalization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 1_000_000)

train_df.dtypes
def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()



add_travel_vector_features(train_df)
print(train_df.isnull().sum())
print('Tamanho antes: %d' % len(train_df))

train_df = train_df.dropna(how = 'any', axis = 'rows')

print('Após exclusão: %d' % len(train_df))
print('Tamanho antes do tratamento: %d' % len(train_df))

train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]

train_df = train_df[(train_df.abs_diff_longitude != 0) & (train_df.abs_diff_latitude != 0)]

dropped_columns = ['pickup_longitude', 'pickup_latitude', 

                   'dropoff_longitude', 'dropoff_latitude']

train = train_df.drop(dropped_columns, axis=1)

print('Tratamento após o tratamento: %d' % len(train))
def late_night (row):

    if (row['hour'] <= 6) or (row['hour'] >= 20):

        return 1

    else:

        return 0





def night (row):

    if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):

        return 1

    else:

        return 0

    

    

def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):

    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)





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

    # Drop 'pickup_datetime' as we won't need it anymore

    df = df.drop('pickup_datetime', axis=1)

    

    return df



train = add_time_features(train)
train.head(5)
dropped_columns2 = ['key', 'passenger_count']

train = train.drop(dropped_columns2, axis=1)

train.head(5)
train_df, validation_df = train_test_split(train, test_size=0.10, random_state=1)



# Get labels

train_labels = train_df['fare_amount'].values

validation_labels = validation_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)

validation_df = validation_df.drop(['fare_amount'], axis=1)
print(len(train_df))

print(len(validation_df))
scaler = preprocessing.MinMaxScaler()

train_df_scaled = scaler.fit_transform(train_df)

validation_df_scaled = scaler.transform(validation_df)

test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')

print(len(test))

#vamos limpar um pouco do dataset

add_travel_vector_features(test)

test = test.drop(dropped_columns, axis=1)



test_aux = test.drop(['passenger_count'], axis=1)

test = test.drop(dropped_columns2, axis=1)



# vamos agora adicionar a feature criada para datas

test = add_time_features(test)



test_df_scaled = scaler.transform(test)



test.head(5)
# Model parameters

BATCH_SIZE = 512

EPOCHS = 20

LEARNING_RATE = 0.001

DATASET_SIZE = 1000000



model = Sequential()

model.add(Dense(256, activation='relu', input_dim=train_df_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1))



sgd = optimizers.SGD(lr=LEARNING_RATE, clipvalue=0.5)

#adam = optimizers.adam(lr=LEARNING_RATE)

model.compile(loss='mse', optimizer=sgd, metrics=['mae'])
history = model.fit(x=train_df_scaled, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, 

                    verbose=1, validation_data=(validation_df_scaled, validation_labels), 

                    shuffle=True)
prediction = model.predict(test_df_scaled, batch_size=128, verbose=1)



print(prediction)

print(os.listdir('.'))