# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from keras_tqdm import TQDMNotebookCallback
#features from basic linear model kernel
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
filename = 'train.csv'
dfs = []
chunksize = 10 ** 6
for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize)):
    #preprocessing section
    add_travel_vector_features(chunk)
    chunk = chunk.dropna(how = 'any', axis = 'rows')
    chunk = chunk[(chunk.abs_diff_longitude < 5.0) & (chunk.abs_diff_latitude < 5.0)]
    chunk = chunk[(chunk.passenger_count > 0) & (chunk.passenger_count <= 6)]
    chunk[['date','time','timezone']] = chunk['pickup_datetime'].str.split(expand=True)
    chunk[['year','month','day']] = chunk['date'].str.split('-',expand=True).astype('int64')
    chunk[['hour','minute','second']] = chunk['time'].str.split(':',expand=True).astype('int64')
    chunk['year_after_0'] = chunk['year'] - np.min(chunk['year'])
    chunk[['trash', 'order_no']] = chunk['key'].str.split('.',expand=True)
    chunk['order_no'] = chunk['order_no'].astype('int64')
    chunk = pd.concat([chunk,pd.get_dummies(chunk['passenger_count'],prefix='pass')], axis =1)
    chunk = chunk.drop(['timezone','date','time', 'pickup_datetime','trash','key','passenger_count'], axis = 1)
    #append chunk to the list
    dfs.append(chunk)
%%time
#concatenate all chunk in one big-ass DataFrame
train_df = pd.concat(dfs)
#delete the chunks as I only have 16 GB RAM
del dfs
train_df.head()
train_df.shape
X_train = train_df.drop(['fare_amount'],axis=1)
Y_train = train_df['fare_amount']
del train_df
scaler = StandardScaler()
y_scaler = StandardScaler()
#scale the data so that columns have zero mean and unit variance
train = scaler.fit_transform(X_train.values)
y_train =  y_scaler.fit_transform(Y_train.values.reshape(-1,1))
del X_train
del Y_train
import keras
import tensorflow as tf
#some imports are unnecessary
from keras import layers
from keras.layers import Input, Dropout,Dense, Activation, BatchNormalization
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras.regularizers import l2
from keras.optimizers import Adam
model = keras.Sequential([
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dense(1, activation=tf.nn.leaky_relu)
])
model.compile(optimizer=Adam(5e-4), 
              loss='mean_squared_error')
filepath = './model_weights/weights-improvement-55M-{epoch:02d}-{val_loss:.4f}.hdf5'
best_callback = ModelCheckpoint(filepath, 
                                save_best_only=True)
lr_sched = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 5, verbose = 1)
tqdm_callback = TQDMNotebookCallback(leave_inner=True,metric_format="{name}: {value:0.5f}")
history = model.fit(train, y_train, 
          epochs=20,
          verbose=0,
          batch_size=2048,
          validation_split=0.0002,
          callbacks=[tqdm_callback,best_callback, lr_sched])
model.load_weights('./model_weights/weights-improvement-55M-19-0.0471.hdf5')
test_df = pd.read_csv('test.csv')
test_df.dtypes
key = test_df.key
add_travel_vector_features(test_df)
test_df[['date','time','timezone']] = test_df['pickup_datetime'].str.split(expand=True)
test_df[['year','month','day']] = test_df['date'].str.split('-',expand=True).astype('int64')
test_df[['hour','minute','second']] = test_df['time'].str.split(':',expand=True).astype('int64')
test_df['year_after_0'] = test_df['year'] - np.min(test_df['year'])
test_df[['trash', 'order_no']] = test_df['key'].str.split('.',expand=True)
test_df['order_no'] = test_df['order_no'].astype('int64')
test_df = pd.concat([test_df,pd.get_dummies(test_df['passenger_count'],prefix='pass')], axis =1)
test_df = test_df.drop(['timezone','date','time', 'pickup_datetime','trash','key','passenger_count'], axis = 1)
# Predict fare_amount on the test set using our model (w) tested on the testing set.
test_df.shape
test = scaler.transform(test_df.values)
y_test = model.predict(test)
y_test = y_scaler.inverse_transform(y_test).reshape(-1)
# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': key, 'fare_amount': y_test},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_100.csv', index = False)

print(os.listdir('.'))
