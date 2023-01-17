# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



%matplotlib inline
# Load the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Train dataset has 'label' in the first column

# Split it and make X and Y

Y_train_orig = train['label']

X_train_orig = train.drop(labels=['label'], axis=1)

print(f'Shape of X is: {X_train_orig.shape}')

print(f'Shape of Y is: {Y_train_orig.shape}')
# Standardize dataset. Divide by 255 to get all x values between 0 and 1

X_train = X_train_orig / 255.

test = test / 255.



# Convert Y to one-hot vectors

Y_train = pd.get_dummies(Y_train_orig)



# Reshape X to dimensions (m, n_h, n_w, n_c)

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



print(X_train.shape)

print(Y_train.shape)



X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)



# X_train, X_dev, Y_train, Y_dev = X_train.T, X_dev.T, Y_train.T, Y_dev.T

print(X_train.shape)

print(Y_train.shape)
index = 560

plt.figure()

plt.imshow(X_train[index][:,:,0])

plt.colorbar()

plt.grid(False)

print ("y = " + str(np.squeeze(Y_train.values[index, :])))
def model(input_shape):

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!

    X_input = Input(input_shape)



    # Zero-Padding: pads the border of X_input with zeroes

    X = ZeroPadding2D((3, 3))(X_input)



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    

    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(64, (7, 7), strides = (1, 1), name = 'conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool1')(X)



    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(128, activation='relu', name='fc0')(X)

    X = Dense(128, activation='relu', name='fc1')(X)

    X = Dense(10, activation='softmax', name='fc2')(X)



    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs = X_input, outputs = X, name='DigitRecognizer')



    return model
datagen = ImageDataGenerator(

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1, 

    horizontal_flip=False,

    vertical_flip=False)



datagen.fit(X_train)
# Trying some things for improving performance

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
print(X_train.shape[1:])

digit_recognizer = model(X_train.shape[1:])

digit_recognizer.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# digit_recognizer.fit(x = X_train, y = Y_train, epochs = 5, batch_size = 16, validation_data=(X_dev, Y_dev))

digit_recognizer.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), 

                               epochs=30, 

                               steps_per_epoch=len(X_train) / 32, 

                               validation_data=(X_dev, Y_dev), 

                               callbacks=[reduce_lr])
# predict results

results = digit_recognizer.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit_recognizer_cnn.csv",index=False)