import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split
train_ds = pd.read_csv('../input/train.csv')

test_ds = pd.read_csv('../input/test.csv')

print(train_ds.shape, test_ds.shape)
train_y = train_ds['target']

train_x = train_ds.drop(['target', 'ID_code'], axis=1)

id_test = test_ds['ID_code']

test_x = test_ds.drop(['ID_code'], axis=1)



train_x = scale(train_x)

test_x = scale(test_x)



train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.45, random_state=82)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005), input_shape=(200,)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x = train_x, y = train_y, batch_size = 25800, epochs=20, validation_data=(dev_x, dev_y))
prediction = model.predict(test_x)

pd.DataFrame({"ID_code":id_test,"target":prediction[:,0]}).to_csv('result_keras.csv',index=False,header=True)