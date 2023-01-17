# The problem we will solve is to convert from Celsius to Fahrenheit

# where the approximate formula is: f=c×1.8+32 



# Of course, it would be simple enough to create a conventional Python function 

# But that wouldn't be machine learning.



# Instead, we will give TensorFlow some sample Celsius values (0, 8, 15, 22, 38) and 

# their corresponding Fahrenheit values (32, 46, 59, 72, 100). 

# Then, we will train a model that figures out the above formula through the training process.
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
c = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)

f = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
# Feature — The input(s) to our model. 

#           In this case, a single value — the degrees in Celsius.

# Labels —  The output our model predicts. 

#           In this case, a single value — the degrees in Fahrenheit.

# Example — A pair of inputs/outputs used during training. 

#           In our case a pair of values from celsius_q and fahrenheit_a at a specific index, such as (22,72).
# Since the problem is straightforward, 

# this network will require only a single layer, 

# with a single neuron.
model = tf.keras.Sequential([

  tf.keras.layers.Dense(units=1, input_shape=[1])

])
# model definition takes a list of layers as argument, 

# specifying the calculation order from the input to the output.



# input_shape=[1] — This specifies that the input to this layer is a single value. 

#                   That is, the shape is a one-dimensional array with one member. 



# units=1 — This specifies the number of neurons in the layer. 

#           The number of neurons defines how many internal variables the layer has to try to learn how to solve the problem. 
model.compile(loss='mean_squared_error',

              optimizer=tf.keras.optimizers.Adam(0.1), metrics=['mean_squared_error'])
# Loss function — A way of measuring how far off predictions are from the desired outcome. 

#                 (The measured difference is called the "loss".)



# Optimizer function — A way of adjusting internal values in order to reduce the loss.
history = model.fit(c, f, epochs=500, verbose=False)

print("Finished training the model")
history.params
plt.xlabel('Epoch Number')

plt.ylabel("Loss Magnitude")

plt.plot(history.history['loss'])

plt.show()
print(model.predict([100.0]))
# f = c×1.8+32 



f_by_hand = (100*1.8)+32

print(f_by_hand) 
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])

l1 = tf.keras.layers.Dense(units=4)

l2 = tf.keras.layers.Dense(units=1)



model = tf.keras.Sequential([l0, l1, l2])



model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['mean_squared_error'])



history = model.fit(c, f, epochs=100, verbose=False)

print("Finished training the model")



print(model.predict([100.0]))



plt.xlabel('Epoch Number')

plt.ylabel("Loss Magnitude")

plt.plot(history.history['loss'])