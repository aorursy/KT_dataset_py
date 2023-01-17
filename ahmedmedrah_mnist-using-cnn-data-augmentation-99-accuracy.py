# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

# Set seed

np.random.seed(0)
# Load data and target from MNIST data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

print(X_train.shape)



del train


# Reshape training image data into features

X_train = X_train.values.reshape(-1,28,28,1)

# Reshape test image data into features

test = test.values.reshape(-1,28,28,1)

# Rescale pixel intensity to between 0 and 1

X_train = X_train / 255.0

test = test / 255.0

# One-hot encode target

Y_train = to_categorical(Y_train, num_classes = 10)

number_of_classes = Y_train.shape[1]
random_seed = 42



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed) # Splitting our data
# Start neural network

network = Sequential()

# Add convolutional layer with 32 filters, a 3x3 window, and ReLU activation function

network.add(Conv2D(filters=32,

            kernel_size=(3, 3),

            input_shape=(28, 28, 1),

            activation='relu'))

# Add max pooling layer with a 2x2 window

network.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout layer

network.add(Dropout(0.3))

# Add convolutional layer with 64 filters, a 3x3 window, and ReLU activation function

network.add(Conv2D(filters=64,

            kernel_size=(3, 3),

            input_shape=(28, 28, 1),

            activation='relu'))

# Add max pooling layer with a 2x2 window

network.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout layer

network.add(Dropout(0.3))

# Add layer to flatten input

network.add(Flatten())

# Add fully connected layer of 128 units with a ReLU activation function

network.add(Dense(128, activation="relu"))

# Add dropout layer

network.add(Dropout(0.3))

# Add fully connected layer of 128 units with a ReLU activation function

network.add(Dense(128, activation="relu"))

# Add dropout layer

network.add(Dropout(0.3))

# Add fully connected layer with a softmax activation function

network.add(Dense(number_of_classes, activation="softmax"))

# Compile neural network

network.compile(loss="categorical_crossentropy", # Cross-entropy

                optimizer="adam", # Root Mean Square Propagation

                metrics=["accuracy"]) # Accuracy performance metric
datagen = ImageDataGenerator( 

    rotation_range=10, 

    zoom_range = 0.1, 

    width_shift_range=0.1,  

    height_shift_range=0.1,  

    horizontal_flip=False,  

    vertical_flip=False)  





datagen.fit(X_train)
epochs = 10

batch_size = 128



history = network.fit_generator(datagen.flow(X_train,Y_train, batch_size=128),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best')



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best')
results=network.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist.csv",index=False)