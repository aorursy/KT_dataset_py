# Write code here that prints the output of the above neuron
print (1*2 + 3*4 + 4*5)
# Import some libraries

import numpy as np
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
dataset = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]])

X = dataset[:, :2]
y = dataset[:, -1]
# Trivial exercise, print the values of X, print the values of y. 
# Confirm that you understand how are we slicing the dataset into X's and y's.
# Confirm that the X's and y's are the ones from the table above by doing the following exercise, assign into the variable 'xyz' all the elements from the second row.
# Print the "shape" of X and y, what does shape mean?

# print the values of X
print (X)

# print the values of y

# assign into the variable 'xyz' all the elements from the second row
xyz = None

# print the shape of X and y, what does shape mean?
print (X.shape)


# A Neural Network is a sequence of layers. 
# In Keras, you define one by instantiating "Sequential"
# https://keras.io/models/sequential/

model = Sequential()
model.add(Dense(3, input_dim=2, activation='relu')) # This is our hidden layer
model.add(Dense(1, activation='sigmoid')) # This is our output layer

# What the above is saying is that we are adding two layers (three actually...).
# The first layer is our hidden layer with 3 nodes. We also specify that it has an "input_dim" 
# or input dimension of 2 (our two inputs).

# The next layer is our output layer with just one node.
# We will worry later about the activation functions.
# We then proceed to compile our model. 
# Two required parameters are the 'loss' and an 'optimizer'.

# 'loss' is the objective function. Suffice to say for now that when the network
# has just one output and it's expected to have two values (like true/false or 0/1)
# we can use 'binary_crossentropy'
# Later on we will use another objective function when we have more than one output.

# We will not dive deeper into the 'optimizer' and stick to use 'adam'.
model.compile(loss='binary_crossentropy', optimizer='adam')
# We can now do the training, Keras calls it 'fit', as in, fit our model to the X -> y mapping.

# batch_size is how many samples to do within a training epoch.
# We don't need to dive deeper into it.
# We only have 4 elements in our dataset, so let's use a batch_size of 4.

# 'epochs' is the number of training iterations to run. This is an important parameter.
# The more iterations we do, the better our model will map the X to the y's that we give.
# However, notice that that means that we are mapping the training set, having a really high
# number of epochs can cause 'overfit' !.
# In this particular case, we don't have a test set, so we don't care.
model.fit(X, y, batch_size=4, epochs=5000)
print (model.predict_classes(X))
print (model.predict(X))
print (model.summary())
print (model.get_weights())
# Let's read our training and test set from the attached dataset
train = pd.read_csv("../input/train.csv").values

# The first column of the dataset tell us the actual value
x_train = train[:, 1:]
y_train = train[:, 0]

# So far we have loaded the 2 variables that we need
# 1. x_train -> has the images to train on
# 2. y_train -> has the value for those images
# How many images do we have?
print (x_train.shape)
print (x_train[0])
# Let's visualize the first ten images.

fig1, ax1 = plt.subplots(1,12, figsize=(12,10))
for i in range(12):
    ax1[i].imshow(x_train[i].reshape((28,28)), cmap='gray')
    ax1[i].axis('off')
    ax1[i].set_title(y_train[i])
# Exercise, visualize the last figure in the set, what number does it contain?

# TODO
num_pixels = x_train.shape[1]

print (num_pixels)
# Let's normalize from 0 to 255 to 0 to 1. It's a good practice when dealing with 
# neural networks that the inputs have short range of values.
x_train = x_train.astype('float32')
x_train /= 255

print (x_train[0])
# Now, the "y_train" is an array of values from 0 to 9.
# Let's print the first 10.
print (y_train[:10])
# What this means for us is that we need to transform our y_train from 0-9 to a 'one-hot encoding'
print ('Before')
print (y_train[:10])

y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

print ('After one hot encoding')
print (y_train[:10])
# We're finally ready to construct our model.
# It's practically identical as the one for the XOR.

# The few differences are:
# 1. We are using a 'softmax' activation for the output layer.
# 2. We are using a 'categorical_crossentropy' loss function.
model = Sequential()

model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print (model.summary())
model.fit(x_train, y_train, epochs=10, batch_size=200, verbose=2, validation_split=0.2)