import pandas as pd

import numpy as np



from random import random, sample, seed
train = pd.read_csv('../input/train.csv',infer_datetime_format=True,parse_dates=["pickup_datetime"])

print(train.shape)
## drop outlier duration trips. I leave in 0 passenger trips and the like, so you may want to clean differently



duration_mask = ((train.trip_duration < 70) | # < 1.1 min

             (train.trip_duration > 3600*4)) # > 4 hours # orig: 3,600 = 1 hours

print('Anomalies in trip duration, %: {:.2f}'.format(

    train[duration_mask].shape[0] / train.shape[0] * 100

))

train = train[~duration_mask] # drop 10k anomalies

print(train.shape)
train.head()
train.head().pickup_datetime
# seconds since start of day

train["seconds_elapsed"] = (train.pickup_datetime - train.pickup_datetime.dt.normalize()).dt.seconds



## Add cyclical time features 



# train['week_delta_sin'] = np.sin((train["pickup_datetime"].dt.dayofweek / 7) * np.pi)**2

# train['hour_sin'] = np.sin((train["pickup_datetime"].dt.hour / 24) * np.pi)**2
hours = np.array(train["pickup_datetime"].dt.hour, dtype=int)

minutes = np.array(train["pickup_datetime"].dt.minute, dtype=int)

dayofweeks = np.array(train["pickup_datetime"].dt.dayofweek, dtype=int)

dayofyear = np.array(train["pickup_datetime"].dt.dayofyear, dtype=int)
print(hours[0:2])

print(minutes[0:2])

print(dayofweeks[0:2])

print(dayofyear[0:2])
dayofyears_tf = dayofyear - 1



print(dayofyears_tf[0:10])
from keras.models import Input, Model

from keras.layers import Dense, Embedding, GlobalAveragePooling1D, concatenate, Activation

from keras.layers.core import Masking, Dropout, Reshape

from keras.layers.normalization import BatchNormalization



batch_size = 64

embedding_dims = 64

epochs = 20
meta_embedding_dims = 64



hours_input = Input(shape=(1,), name='hours_input')

hours_embedding = Embedding(24, meta_embedding_dims)(hours_input)

hours_reshape = Reshape((meta_embedding_dims,))(hours_embedding)



dayofweeks_input = Input(shape=(1,), name='dayofweeks_input')

dayofweeks_embedding = Embedding(7, meta_embedding_dims)(dayofweeks_input)

dayofweeks_reshape = Reshape((meta_embedding_dims,))(dayofweeks_embedding)



minutes_input = Input(shape=(1,), name='minutes_input')

minutes_embedding = Embedding(60, meta_embedding_dims)(minutes_input)

minutes_reshape = Reshape((meta_embedding_dims,))(minutes_embedding)



dayofyears_input = Input(shape=(1,), name='dayofyears_input')

dayofyears_embedding = Embedding(366, meta_embedding_dims)(dayofyears_input)

dayofyears_reshape = Reshape((meta_embedding_dims,))(dayofyears_embedding)
merged = concatenate([ hours_reshape, dayofweeks_reshape, minutes_reshape, dayofyears_reshape])



hidden_1 = Dense(256, activation='relu')(merged)

hidden_1 = BatchNormalization()(hidden_1)



main_output = Dense(1, activation='sigmoid', name='main_out')(hidden_1)





model = Model(inputs=[hours_input,

                      dayofweeks_input,

                      minutes_input,

                      dayofyears_input], outputs=[main_output])



model.compile(loss='mean_squared_error', optimizer='adam')



model.summary()