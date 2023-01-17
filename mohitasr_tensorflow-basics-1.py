import numpy as np 

import tensorflow as tf

from tensorflow import keras



print("Imported the necessary libraries...")
# First we define the Neural Network structure. This NN a single layer,1 neuron, 1 input value

my_first_model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])





# Next, we compile the model that we just created

my_first_model.compile(optimizer='sgd', loss='mean_squared_error')



print("Defined and compiled the model...")
x_train = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

y_train = np.array([-17.0, -12.0, -7.0, -2.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0])



print("Training data created...")
my_first_model.fit(x_train, y_train, epochs=500)
my_first_model.predict([80])