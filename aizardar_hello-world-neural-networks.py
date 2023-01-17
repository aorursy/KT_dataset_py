import tensorflow as tf
import numpy as np
from tensorflow import keras # We use keras API to define our model
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])]) 

model.compile(optimizer = 'sgd', loss = 'mean_squared_error') # 'sgd' stands for stochastic gradient descent
# Time to provide the data to our NN

X = np.array([-1,0,1,2,3,4,5,6,7,8,9,10], dtype = float)
Y = np.array([-10,0,10,20,30,40,50,60,70,80,90,100], dtype = float)

# Training the NN

model.fit(X,Y,epochs = 500)
print(model.predict([25.0]))