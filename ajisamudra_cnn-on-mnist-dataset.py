# Library



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import datetime



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split 



import warnings

warnings.filterwarnings('ignore')
# Read file

df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



X_train = df_train.iloc[:,1:]

y_train = df_train.iloc[:,0]



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,

                                                    stratify = y_train,

                                                    test_size=0.2, random_state = 41)



X_train.shape, y_train.shape, X_val.shape, y_val.shape
# Prepare dataset for CNN



# Get data from previous model

arr_X_train = X_train.to_numpy()

arr_X_val = X_val.to_numpy()

arr_y_train = y_train.to_numpy()

arr_y_val = y_val.to_numpy()



# Get data from csv file and normalize it

arr_X_test = df_test.to_numpy()



# Normalize data

arr_X_train = arr_X_train / 255

arr_X_val = arr_X_val / 255

arr_X_test = arr_X_test / 255



# input image dimensions

img_rows, img_cols = 28, 28

num_classes = 10



# Reshape the array to (28,28,1)

X_train = arr_X_train.reshape(arr_X_train.shape[0], img_rows, img_cols, 1)

X_val = arr_X_val.reshape(arr_X_val.shape[0], img_rows, img_cols, 1)

X_test = arr_X_test.reshape(arr_X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_val = keras.utils.to_categorical(y_val, num_classes)



X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape
# Define CNN architecture



# This is my previous CNN architecture

# The model is constructed by several layers

# I use relu activation and he initializer to get faster converge time

# The first layer is convolutional layer with 16 filter, 3x3 kernel size, 1 stride

# The second layer is the same with the first layer. They are supposed to capture feature maps from digit image.

# The third layer is pooling with maximum aggregation. It is used to reduce the size of feature maps.

# The fourth layer is convolutional layer for the result from maximum pooling.

# The next layer is dense neural network with 256 units, before feeding this layer with data I need to flatten first the data

# The last layer / output layer is dense neural network with 10 units (the same with number of class)



cnn_model = keras.models.Sequential([

    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer= 'he_normal'),

    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer= 'he_normal'),

    keras.layers.MaxPooling2D(pool_size=(3, 3)),

    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer= 'he_normal'),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu', kernel_initializer= 'he_normal'),

    keras.layers.Dense(num_classes, activation='softmax')

])



# Compile the model

cnn_model.compile(optimizer= 'adam',

                  loss=keras.losses.categorical_crossentropy,

                  metrics= ['accuracy']

                 )
# Train the model

cnn_model.fit(X_train, y_train, epochs=3)
## Evaluate model in validation dataset

test_loss, test_accuracy = cnn_model.evaluate(X_val, y_val, verbose= 0)

print("CNN Accuracy on Validation: {}".format(test_accuracy))
# Function to train and evaluate

def train_and_validate(model, epochs):

    model.fit(X_train, y_train, epochs=epochs)

    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose= 0)

    print("CNN Accuracy on Validation: {}".format(test_accuracy))
# Challenger architecture



cnn2_model = keras.models.Sequential([

    keras.layers.Conv2D(64, 7, input_shape=input_shape, activation='relu',  padding='same', kernel_initializer= 'he_normal'),

    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(128, 3, activation='relu',  padding='same', kernel_initializer= 'he_normal'),

    keras.layers.Conv2D(128, 3, activation='relu',  padding='same', kernel_initializer= 'he_normal'),

    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(256, 3, activation='relu',  padding='same', kernel_initializer= 'he_normal'),

    keras.layers.Conv2D(256, 3, activation='relu',  padding='same', kernel_initializer= 'he_normal'),

    keras.layers.MaxPooling2D(2),    

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu', kernel_initializer= 'he_normal'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(64, activation='relu', kernel_initializer= 'he_normal'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes, activation='softmax')

])



# Compile the model

cnn2_model.compile(optimizer= 'adam',

                  loss=keras.losses.categorical_crossentropy,

                  metrics= ['accuracy']

                 )
# Train and evaluate 

train_and_validate(cnn2_model, 12)
# Prediction for submission

arr_y_pred = cnn2_model.predict(X_test)



# Create list of prediction

y_pred = []

for i in range(len(arr_y_pred)):

    y_pred.append(np.argmax(arr_y_pred[i]))



# Create file submisssion from CNN

submission_cnn = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

submission_cnn.iloc[:,1] = (y_pred)

submission_cnn.to_csv("submission_cnn2.csv", index=False)