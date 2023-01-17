import tensorflow as tf

import numpy as np

from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([2.0,3.0,4.0,5.0,6.0,8.0,9.0,10.0],dtype=float)

ys = np.array([150.0,200.0,250.0,300.0,350.0,450.0,500.0,550.0],dtype=float)

model.fit(xs,ys,epochs=500)

print(model.predict([7.0]))