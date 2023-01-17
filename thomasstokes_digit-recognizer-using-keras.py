# This Python 3 enle competitions download -c digit-recognizervironment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# Here are all the packages that I will be using

import numpy as np # Linear algebra

import pandas as pd # Data processing 

import matplotlib.pyplot as plt # Data visualistaion

from sklearn.model_selection import train_test_split # Creating the training for the neural network

from keras.models import Sequential # The structure of the neural network

from keras import backend as K # Output layer of the neural network

from keras.layers import Dense , Dropout , Lambda, Flatten # Functions for the inner layers of the neural network

from keras.optimizers import Adam ,RMSprop # Optimizing the network 

from keras.preprocessing.image import ImageDataGenerator # Preprocessing the images



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Let's look at the training data

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(train.shape)

train.head()
# Let's get an overview of the training data

train.info()
# Let's now look at the test data

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.shape) 

test.head()
# Let's get an overview of the test data

test.info()
# Now I want to split up the train data into the images (X) and the labels (y).

X_train = (train.iloc[:,1:].values).astype('float32') # Images (represented as pixel values)

y_train = train.iloc[:,0].values.astype('int32') # Labels (numbers represented by the images)

X_test = test.values.astype('float32')
# View X_train

X_train
# View y_train

y_train
# Reshape the data so that into a 28x28 grid so that the pixels form the corresponding original image

X_train = X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_test = X_test.reshape(X_test.shape[0], 28, 28)



# Generate the images

for i in range(20,23):

    plt.figure(figsize=(14,14)) # Scale up the image (to make it easier to see)

    plt.subplot(500 + (i+1)) # Creates the suplot

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray')) # Displays the image using a greyscale colour map

    plt.title(y_train[i]) # Title for our image
# Reshaping the data so it adds in a greyscale dimension

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train.shape
# Normalising the training data

# Normalising formula: Z = (X-mean)/std, this changes the distrubtion of the data to N~(mean=0, var=1)

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
from keras.utils.np_utils import to_categorical # Changes the data to categorical data (the type we want)

y_train= to_categorical(y_train)

num_classes = y_train.shape[1] # Number to columns in our new y vector

num_classes
# Fix the random seed for reproducibility

seed = 37

np.random.seed(seed)
from keras.models import  Sequential # Linear model

from keras.layers.core import  Lambda , Dense, Flatten, Dropout # Functions for manipulating the inner layers of the neural network

from keras.callbacks import EarlyStopping # Checking the models accuracy whilst training

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D # Inner layers of the neural network
model= Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(10, activation='softmax')) # softmax function converts the inputs to a vector with elements in (0,1) with the sum of the elements = 1

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)
from keras.optimizers import RMSprop # RMSprop uses the moving mean squared average of the gradients to optimise the model 

model.compile(optimizer=RMSprop(lr=0.001), # lr is the learning rate (default is 0.001)

 loss='categorical_crossentropy', # categorical_crossentropy is used for labels that are one-hot vectors

 metrics=['accuracy']) # Calculates how often the neural networks prediction matches the label
from keras.preprocessing import image

gen = image.ImageDataGenerator() # Generates batches of tensor image data 
from sklearn.model_selection import train_test_split # Splits the training data to help avoid overfitting

X = X_train

y = y_train

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=37)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches=gen.flow(X_val, y_val, batch_size=64)
# Train the model

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# Creates a Fully connected model

def get_fc_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Flatten(),

        Dense(512, activation='relu'), # relu (rectified linear unit) is the max(x,0), https://keras.io/api/layers/activations/#relu-function

        Dense(10, activation='softmax')

        ])

    model.compile(optimizer='Adam', # Adam is also known as Stochastic gradient descent, https://keras.io/api/optimizers/adam/ 

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
# Train the model

fc = get_fc_model()

fc.optimizer.lr=0.01

history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# Creating the CNN

from keras.layers import Convolution2D, MaxPooling2D



def get_cnn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),# Convolution2D(filters (specifies the number of pieces the data is divided into), strides (how many neighbouring inputs the layer considers), activation)

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(), # Compacts the layer by taking the max value in a given window (default is 2x2)

        Convolution2D(64,(3,3), activation='relu'),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        Dense(512, activation='relu'),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), 

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
# Train the CNN

model= get_cnn_model()

model.optimizer.lr=0.01

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# Augment the data

gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.09, shear_range=0.3,

                               height_shift_range=0.09, zoom_range=0.04)

batches = gen.flow(X_train, y_train, batch_size=64) # Create the batches

val_batches = gen.flow(X_val, y_val, batch_size=64) # Create the labels for the batches
# Train the model with the augmented data

model.optimizer.lr=0.001

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# Create the BN model

from keras.layers.normalization import BatchNormalization



def get_bn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(axis=1),

        Convolution2D(32,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
# Update the Model with BN

model= get_bn_model()

model.optimizer.lr=0.01

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
# Train the model on the full dataset

model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X, y, batch_size=64)

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1)
# Kaggle Predictions

predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DRC.csv", index=False, header=True)