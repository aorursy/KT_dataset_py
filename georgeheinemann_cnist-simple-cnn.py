# Suppress multiple warnings

import warnings

warnings.filterwarnings(action='once')
# Import libraries and modules

import numpy as np

import pandas as pd

import os

import datetime
# Files in directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import train

Train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(Train.shape)



# Import test

X_test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(X_test.shape)



# Quick look at the data

Train.head()
# Split train into X & Y

y_train = Train.label

X_train = Train.drop('label',axis=1)
# Convert all into numpy arrays

y_train = y_train.values

X_train = X_train.values

X_test = X_test.values
# Split train data into t_train and t_test sets

from sklearn.model_selection import train_test_split

Xt_train, Xt_test, yt_train, yt_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
# Return shape of 1st item in list

Xt_train[0].shape
# Visualise data



import matplotlib.pyplot as plt 

data_no = 6

plt.imshow(Xt_train[data_no].reshape(28, 28))

plt.title('Label: ' + str(yt_train[data_no]))
# Preprocess input data



# Good practice (and better performance) to ensure the values are less than 1. To do this we'll calculate the max value and scale all numbers by that.

scaling_value = max(Xt_train.max().max(),Xt_test.max().max(),X_test.max().max())

Xt_train = Xt_train/ scaling_value

Xt_test = Xt_test/ scaling_value

X_test = X_test/ scaling_value



# Convert inputs into right form for model shape (n, width, height) to (n, depth, width, height), where n is number of records.

Xt_train = Xt_train.reshape(Xt_train.shape[0], 28, 28, 1)

Xt_test = Xt_test.reshape(Xt_test.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Ensure inputs are correct type

Xt_train = Xt_train.astype('float32')

Xt_test = Xt_test.astype('float32')

X_test = X_test.astype('float32')



# Preprocess class labels (we require these to be one-hot encoded i.e. for 3 [0,0,1,0,0,0,0,0,0])

from keras.utils import np_utils

yt_train = np_utils.to_categorical(yt_train, 10)

yt_test = np_utils.to_categorical(yt_test, 10)



# Sense check some parameters

print('Scaling value: ' + str(scaling_value))

print(Xt_train.shape)

print(yt_train.shape)
from keras.models import Sequential



# Define model architecture

model = Sequential() 
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D



# Add layers



# The objective of the Convolution Operation is to extract the high-level features such as edges, from the input image.

# A 3x3 kernal (or filter) is passed across the image

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

print(model.output_shape)

model.add(Conv2D(32, kernel_size=3, activation='relu'))

print(model.output_shape)



# MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter

# across the previous layer and taking the max of the 4 values in the 2x2 filter

model.add(MaxPooling2D(pool_size=(2,2)))

print(model.output_shape)



# Dropout is a method for regularizing our model in order to prevent overfitting.

model.add(Dropout(0.25))

print(model.output_shape)



# Deep network requires to be converted into a 1-D array

model.add(Flatten())

print(model.output_shape)



# Add a couple of fully connected dense layers

model.add(Dense(128, activation='relu'))

print(model.output_shape)

# Last layer is out output so a 'softmax' outputs a probability distribution between 0 and 1

model.add(Dense(10, activation='softmax'))

print(model.output_shape)
# Compile model

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# Fit model on training data

start = datetime.datetime.now()

model.fit(Xt_train, yt_train, batch_size=32, nb_epoch=10, verbose=1)

print('Time taken: ' + str((datetime.datetime.now()-start).total_seconds()) + ' seconds')
# Evaluate model on test data

score = model.evaluate(Xt_test, yt_test, verbose=0)

print(model.metrics_names[0] + ": " + str(score[0]))

print(model.metrics_names[1] + ": " + str(score[1]))
# Fit model on ALL training data

X_train = np.concatenate([Xt_train,Xt_test],axis=0)

y_train = np.concatenate([yt_train,yt_test],axis=0)

start = datetime.datetime.now()

model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1)

print('Time taken: ' + str((datetime.datetime.now()-start).total_seconds()) + ' seconds')
# Predict results on test set

results = model.predict(X_test)



# Select the index with the maximum probability

results = np.argmax(results,axis = 1)



# Put into dataframe and add Id column

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



# Check submission looks correct

submission.head(5)
# Check one result

test_no = 0

plt.imshow(X_test[test_no].reshape(28, 28))

plt.title('Predicted Label: ' + str(submission.Label[test_no]))
# Output to csv for submission

submission.to_csv("submission.csv",index=False)