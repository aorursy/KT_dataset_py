import tensorflow as tf

from tensorflow import keras

import numpy as np
model = tf.keras.Sequential([keras.layers.Dense(units=1 , input_shape=[1])])
model.compile(optimizer='sgd' , loss='mean_squared_error')
noOfBedrooms = np.array([1.0,2.0,3.0,4.0,5.0,6.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0] ,dtype=float)

housePrice = np.array([100.0,150.0,200.0,250.0,300.0,350.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0] , dtype=float)
model.fit(noOfBedrooms , housePrice , epochs=1500)
print((model.predict([7.0])+1) // 100) 