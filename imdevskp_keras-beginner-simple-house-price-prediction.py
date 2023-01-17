# Here we will try to build a neural network that predicts 

# The price of a house according to a simple formula.



# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, 

# So that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.



# How would you create a neural network that learns this relationship 

# so that it would predict a 7 bedroom house as costing close to 400k etc.



# Hint: Your network might work better if you scale the house price down. 

# You don't have to give the answer 400...

# It might be better to create something that predicts the number 4, 

# and then your answer is in the 'hundreds of thousands' etc.
#importing libraries



import numpy as np

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt
model = tf.keras.Sequential([

    keras.layers.Dense(units=1, input_shape=[1])

])



# Sequntial - allows you to create models layer-by-layer for most problems

# Dense - a layer of connected neurons

# units = 1, since there is only one neuron here

# input_shape = [1], since the shape of the array is a 1D array
# We use sgd optimizer which looks at mean squared error to improve the model



model.compile(optimizer='sgd', loss='mean_squared_error')
# A house costs 50k + 50k per bedroom



xs = [1, 2, 3, 4, 5]

ys = [1.0, 1.5, 2.0, 2.5, 3]
# epoch value indicates who many times the model wil go through the training 



history = model.fit(xs, ys, epochs=100)
plt.xlabel('Epoch Number')

plt.ylabel("Loss Magnitude")

plt.plot(history.history['loss'])

plt.show()
# predict the value for the house manually



model.predict([7.0])
# A house costs 50k + 50k per bedroom



print(0.5+7.0*0.5)