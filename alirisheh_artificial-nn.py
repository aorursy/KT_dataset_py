import tensorflow as tf

import numpy as np

import pandas as pd

import keras.backend as K
input_columns = ['year', 'month', 'day', 'hour', 'minute', 'moisture0', 'moisture1', 'moisture2']

prediction = ['moisture3']
train_size = 4000

test_size = 1000
def sum_error(y_true, y_pred):

        return K.sum(K.abs(y_pred - y_true))
K.eval(K.sum(K.abs(np.array([0,3]) - np.array([2,5]))))
data_frame = pd.read_csv('../input/soil-moisture-dataset/plant_vase1.CSV')
data_frame
data_frame = data_frame.sample(frac=1)
data_frame.drop(columns=['irrgation'])
data_frame[['moisture0', 'moisture1','moisture2', 'moisture3']].plot()
X_train = data_frame[input_columns][:train_size]

Y_train = data_frame[prediction][:train_size]

X_test = data_frame[input_columns][train_size:]

Y_test = data_frame[prediction][train_size:]
X_train.shape, Y_train.shape
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=len(input_columns), input_shape=(len(input_columns),), activation='relu'))

model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(units=len(prediction), input_shape=(len(input_columns),)))

model.summary()
model.compile(optimizer='adam', loss=sum_error, metrics=['MSE'])
model.fit(X_train,Y_train, epochs=10)
model.evaluate(X_test, Y_test)
weights = np.array(model.get_weights())
model.get_weights()
Y_test.to_numpy()[58]
prediction = model.predict(X_test.to_numpy())
count = 0

for i in range(len(X_test)):

    if np.abs(prediction[i][0] - Y_test.to_numpy()[i]) > 0.1:

        count += 1

print(count/len(X_test))