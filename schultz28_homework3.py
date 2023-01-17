# Homework Assignment 3 Due: 11/27/2018 11:59PM

# Setup:
# Please review the 'Prepare' https://ben.desire2learn.com/d2l/le/content/333415/viewContent/1740859/View
# 
# Objective: 
# Let's use keras in action.  Design a CNN to categorize 10 classes of clothing types. 
#
# Goal: 
# Train the network to get at least 90% accuracy or higher with tuning.
# Prefix your versions with @V1, @V2, @V3 and place your versions of tuning.
# 
# Information:
# Use the Fashion MNIST (included in Keras) 
# 60k examples of training data
# 10k examples of test data
# 10 classes
	# various types of clothing
	# unique shapes and coloration
#Test loss: 0.2734314037680626
#Test accuracy: 0.9017

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Number of classes
num_classes = 10

# the number of passes through the data
# sizes of the batch and # of epochs of data
batch_size = 128
epochs = 24

# input image dimensions --> size of our image
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# reshape the data for the keras backend understand 
# channel_<'first'><'last'>:  Specifies which data format convention Keras will follow. 
# 1 represents the grayscale --> pass the channels first parameter
# We want to address 'channel_first' issue because this can cause dimension mismatch errors in another backend such as Theano or CNTK
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# scale the pixel intensity from 0-255 to make it fit between 0-1 because the images are grayscale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class metrics using one-hot encoding 
# All categorical data must be converted to numbers in other words the categorical variables is represented as binary vectors.
# 3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
# The first requires that the categorical values be mapped to integer values
# Then, each integer value is represented as a binary vector that is all zero values except the index of the integer which is marked with a 1. 
# This prevents the model from assuming that the numerical class designation is an ordered number
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define a sequential model
model = Sequential()

# 32 filters
# each filter uses a 3 x 3 kernal
# activation function of relu
# pass the first layer of input_shape
model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))

# 2D comes from our images being in two dimensions.
model.add(MaxPooling2D(pool_size=(2,2)))

# repeat the Conv2D and the pool to learn more about the data. 
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Now we know the image features, we can use them to categorize our images into our classes of shirts, sneakers, purses, and etc
# Categorize the classes by using a Dense fully connected layers at the end of the network

# Implement a flatten layer to convert the 2D output of the convolution and max pool into a 1D feature vector we can classify. 
model.add(Flatten())

# Use a dense layer to combine the information
model.add(Dense(128, activation='relu'))

# Add a drop out to avoid the overfitting in the network
model.add(Dropout(0.5))

# Last layer we want to output dense fully connected layer provides probability of the classification
model.add(Dense(num_classes, activation='softmax'))

# Measure loss, which is categorical across entropy and how we are going to optimize to minimize loss. 
# We train the data and evaluate how well each epoch of the training data fits the validation data
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

# Do this using the fit function. 
# Specify the training data, size of the batch, number of epochs, amount of data to display and the data used to validated the training after each epoch. 
# Train the model and test/validate the model with the test data after each cycle (epoch) through the training data
# Return the history of loss and accuracy for each epoch.
# A history dictionary that stores how the model changed as it was trained on the data. 
hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

# Evaluate the model with the test data to get the scores on the 'real' data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Visualize it.

epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
# Homework Assignment 3 Due: 11/27/2018 11:59PM

# @Version 1.0
# Change the Convolutional2D layer from 32 to 64
#Test loss: 0.2630680152595043
#Test accuracy: 0.9105
# Result: The validation loss decreased by around .02 and the accuracy went up by around .01

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 10

batch_size = 128
epochs = 24

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape = input_shape))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
# Homework Assignment 3 Due: 11/27/2018 11:59PM

# @Version 2.0
# Change: Remove an early Pooling layer
# Test loss: 0.26832905543595553
# Test accuracy: 0.9231
# Result: The validation loss went down by around .01 and the accuracy went up by around .02

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 10

batch_size = 128
epochs = 24

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))

# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
# Homework Assignment 3 Due: 11/27/2018 11:59PM

# @Version 3.0
# Change: Add a dropout after each ConV2D layer and add parameters to each Conv2D that include fliters, adding a kernal size and specifying padding
# Also changed the dense later from 64 to 512
# Test loss: 0.21501165002286435
# Test accuracy: 0.9338
# Result: The validation loss went down by around .05 and the accuracy went up by around .03

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 10

batch_size = 128
epochs = 24

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
# Homework Assignment 3 Due: 11/27/2018 11:59PM

# @Version 4.0
# Change: Add another Conv2D Layer and Dense Layer
# Test loss: 0.3659203280316666
# Test accuracy: 0.9271
# Result: The validation loss decreased by around .02 and the accuracy went up by around .01

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 10

batch_size = 128
epochs = 24

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    
model.add(Conv2D(64, (5, 5), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        
model.add(Conv2D(128, (1, 1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        
model.add(Flatten())
    
model.add(Dense(1024, activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
    
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
# Homework Assignment 3 Due: 11/27/2018 11:59PM

# @Version 5.0
# Change: the dense layer from 128 to 512
# Test loss: 0.25888899857997894
# Test accuracy: 0.9161
# Result: The validation loss decreased by around .02 and the accuracy went up by around .01

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 10

batch_size = 128
epochs = 24

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()