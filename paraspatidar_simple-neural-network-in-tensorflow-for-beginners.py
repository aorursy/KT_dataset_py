import tensorflow as tf

import numpy as np
C = [0,4,8,10,15,20,22,30,38,40,50,60,70,80,90,100]

F = [32,39.2,46.4,50,59,68,71.6,86,100.4,104,122,140,158,176,194,212]
model = tf.keras.Sequential([

    tf.keras.layers.Dense(units=1,input_shape=[1]),

    

])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(C,F,epochs=500,verbose=False)

model.predict([125])