import numpy as np

import tensorflow as tf  # Importing the TensorFlow Library

from tensorflow import keras  # Import Keras from TensorFlow
from tensorflow.keras import Sequential

from tensorflow.keras import layers
# Define Sequential model with 3 layers

model = keras.Sequential(

    [

        layers.Dense(2, activation="relu", name="layer1"),

        layers.Dense(3, activation="relu", name="layer2"),

        layers.Dense(4, name="layer3"),

    ]

)

# Call model on a test input

x = tf.ones((3, 3))

y = model(x)

y
# Create 3 layers

layer1 = layers.Dense(2, activation="relu", name="layer1")

layer2 = layers.Dense(3, activation="relu", name="layer2")

layer3 = layers.Dense(4, name="layer3")



# Call layers on a test input

x = tf.ones((3, 3))

y = layer3(layer2(layer1(x)))  # <-- notice how layers are stacked 
model = keras.Sequential(

    [

        layers.Dense(2, activation="relu"),

        layers.Dense(3, activation="relu"),

        layers.Dense(4),

    ]

)
model = keras.Sequential()

model.add(layers.Dense(2, activation="relu"))

model.add(layers.Dense(3, activation="relu"))

model.add(layers.Dense(4))
model = keras.Sequential(name="my_sequential")

model.add(layers.Dense(2, activation="relu", name="layer1"))

model.add(layers.Dense(3, activation="relu", name="layer2"))

model.add(layers.Dense(4, name="layer3"))
# Method 1

model = keras.Sequential()

model.add(keras.Input(shape=(4,))) # Why (4,) and not 4? Because we want our input to be a Vector (1D Tensor)

model.add(layers.Dense(2, activation="relu"))
# Method 2

model = keras.Sequential()

model.add(layers.Dense(2, activation="relu", input_shape=(4,)))  # Combining the 2 code lines into 1
model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model)
# example of a model defined with the sequential api



# define the model

model = Sequential()

model.add(layers.Dense(10, input_shape=(8,)))  

model.add(layers.Dense(1))



model.summary()
# example of a model defined with the sequential api

from tensorflow.keras.layers import Dense



# define the model

model = Sequential()

model.add(Dense(100, input_shape=(8,)))

model.add(Dense(80))

model.add(Dense(30))

model.add(Dense(10))

model.add(Dense(5))

model.add(Dense(1))



model.summary()
plot_model(model)
from tensorflow.keras import Input



# define the layers

x_in = Input(shape=(8,))
dense = layers.Dense(64, activation="relu")

x = layers.Dense(10)(x_in)
x_out = layers.Dense(1)(x)
# example of a model defined with the functional api

from tensorflow.keras import Model



# define the layers

x_in = Input(shape=(8,))

x = Dense(10)(x_in)

x_out = Dense(1)(x)

# define the model

model = Model(inputs=x_in, outputs=x_out)



# print summary

model.summary()
plot_model(model)