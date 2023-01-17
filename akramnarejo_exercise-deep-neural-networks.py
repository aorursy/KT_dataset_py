import tensorflow as tf



# Setup plotting

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')

# Set Matplotlib defaults

plt.rc('figure', autolayout=True)

plt.rc('axes', labelweight='bold', labelsize='large',

       titleweight='bold', titlesize=18, titlepad=10)



# Setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning_intro.ex2 import *
import pandas as pd



concrete = pd.read_csv('../input/dl-course-data/concrete.csv')

concrete.head()
# YOUR CODE HERE

input_shape = [len(concrete.columns)-1]



# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
from tensorflow import keras

from tensorflow.keras import layers



# YOUR CODE HERE

model = keras.Sequential([

    layers.Dense(512, activation='relu', input_shape=input_shape),

    layers.Dense(512, activation='relu'),

    layers.Dense(512, activation='relu'),

    layers.Dense(1),

])



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

q_2.solution()
### YOUR CODE HERE: rewrite this to use activation layers

model = keras.Sequential([

    layers.Dense(units=32,input_shape=[8]),

    layers.Activation('relu'),

    layers.Dense(units=32),

    layers.Activation('relu'),

    layers.Dense(1),

])



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# YOUR CODE HERE: Change 'relu' to 'elu', 'selu', 'swish'... or something else

activation_layer = layers.Activation('relu')



x = tf.linspace(-3.0, 3.0, 100)

y = activation_layer(x) # once created, a layer is callable just like a function



plt.figure(dpi=100)

plt.plot(x, y)

plt.xlim(-3, 3)

plt.xlabel("Input")

plt.ylabel("Output")

plt.show()