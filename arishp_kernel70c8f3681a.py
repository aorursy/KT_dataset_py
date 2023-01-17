# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import seaborn as sns # seabonrn for visualization

import matplotlib.pyplot as plt # plots for visualization

import matplotlib.image as mpimg # to load image data

from sklearn.model_selection import train_test_split # to split train.csv data for cross validation

from sklearn.metrics import confusion_matrix # to construct confusion matrix

import itertools # has iterator functions for efficient looping

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential # to construct in CNN

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # Operations on CNN

from keras.optimizers import RMSprop # RMSProp optimizer

from keras.preprocessing.image import ImageDataGenerator # Image generator class

from keras.callbacks import ReduceLROnPlateau # Reduce learning rate callback function
%matplotlib inline

sns.set()
# Load the data (images stored as pixel values)

train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

test.columns
Y = train["label"] # target of train.csv data

X = train.drop(labels = ["label"],axis = 1) # Drop 'label' column from train.csv data

testX = test.drop(labels = ["id"],axis = 1) # Drop 'id' column from test.csv data

g = sns.countplot(Y) # display the distribution of digits in train.csv data

Y.value_counts().sort_values() # count the distribution of digits in train.csv data
del train # delete train dataframe to free some space 

#del test # delete test dataframe to free some space
#Check for null and missing values in train.csv data

X.isnull().any().describe() # the result shows there is no null values
#Check for null and missing values in test.csv data

testX.isnull().any().describe() # the result shows there is no null values
X = X / 255.0 # Normalize train.csv data

testX = testX / 255.0 # Normalize test.csv the data
# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)

X = X.values.reshape(-1,28,28,1)

testX = testX.values.reshape(-1,28,28,1)
X.shape #reshaped from "42000 X 784" to "60000 X 28 X 28 X 1"
testX.shape
Y = to_categorical(Y, num_classes = 10) # one hot encoding (example: 2 -> [0,0,1,0,0,0,0,0,0,0])
Y # display the encoded target column
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, shuffle=True) # Split the train and the validation set for cross validation
# dimensions of the cross validation data

print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)
g = plt.imshow(X_train[0][:,:,0]) # first image
g = plt.imshow(X_train[1][:,:,0]) # second image
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu',

                 input_shape = (28,28,1))) # input 2D convolutional layer

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2))) # 2D max pooling

model.add(Dropout(0.25)) # applies dropout to prevent overfitting



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten()) # Flattens without affecting the size

model.add(Dense(256, activation = "relu")) # dense layer of size 256 units

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) # dense layer of size 10 to output the digit value
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) # RMSprop optimizer divides the gradient by root mean square

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"]) # Compile the model

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            factor=0.3, 

                                            verbose=1, 

                                            min_lr=0.00001) # reduces learning rate if the learning is stagnant
# Generate batches of tensor image data with real-time data augmentation

# The data will be looped over (in batches).

# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # zoom image

        width_shift_range=0.1,  # shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # shift images vertically (fraction of total height)

        horizontal_flip=False,  # flip images horizontally

        vertical_flip=False)  # flip images vertically

datagen.fit(X_train) # apply datagen augmentation to train.csv data
# Fit the model

epochs = 30

batch_size = 96

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# list all data in history

print(history.history.keys())

#history.history['loss']
fig, ax = plt.subplots(2,1) # aligning two plots horizontally

ax[0].plot(history.history['loss'], label="Training loss") 

ax[0].plot(history.history['val_loss'],label="validation loss")

legend = ax[0].legend(loc='best')



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best')
results = model.predict(testX) # predicting test.csv data

results = np.argmax(results,axis = 1) # select the index with maximum probability

results = pd.Series(results,name="label")
# creating submission file

#submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1) 

submission = pd.concat([test['id'],results],axis = 1) 

submission.to_csv("submission.csv",index=False)

print('csv file ready for submission')