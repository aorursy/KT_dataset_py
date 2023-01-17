# Base packages used

import numpy as np

import keras



# Specific neural network models & layer types

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

import os
print(os.listdir('../input/keras-mnist/'))



def mnist_load_data(path='mnist.npz'):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


# Load the data, split between train and test sets by default

(X_train, y_train), (X_test, y_test) = mnist_load_data(path='../input/keras-mnist/mnist.npz')



# Check out the data

print(f'X_train shape: {X_train.shape}')

print(f'y_train shape: {y_train.shape}')

print(f'X_test shape: {X_test.shape}')

print(f'y_test shape: {y_test.shape}')

print(f'X range: {X_train.min()}-{X_train.max()}')

print(f'y values: {np.unique(y_train)}')

num_classes = len(np.unique(y_train))

print(f'Number of classes: {num_classes}')
# Define input image dimensions

img_rows, img_cols = 28, 28



# Reshape for Keras model types

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)



print(f'X_train shape: {X_train.shape}')

print(f'X_test shape: {X_test.shape}')
# Modify the X values to be 0-1 instead of 0-255

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

print(f'X_train range: {X_train.min()}-{X_train.max()}')
# Modify the y labels from class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

print(f'y_train shape: {y_train.shape}')

print(f'y_test shape: {y_test.shape}')
# Create simple CNN model architecture with Pooling for dimensionality reduction 

# and Dropout to reduce overfitting

CNN_model = Sequential()



CNN_model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = (28, 28, 1)))

CNN_model.add(Conv2D(64, (3, 3), activation='relu'))

CNN_model.add(MaxPooling2D(pool_size=(2, 2)))



CNN_model.add(Dropout(0.25))

CNN_model.add(Flatten())

CNN_model.add(Dense(128, activation='relu'))

CNN_model.add(Dropout(0.5))

CNN_model.add(Dense(num_classes, activation='softmax'))



CNN_model.summary()
# Compile the model with the desired loss function, optimizer, and metric to optimize

CNN_model.compile(loss = 'categorical_crossentropy',

                  optimizer = 'Adam',

                  metrics = ['accuracy'])
# Fit the model on the training data, defining desired batch_size & number of epochs,

# running validation on the test data after each batch

CNN_model.fit(X_train, y_train,

              batch_size = 128,

              epochs = 12,

              verbose = 1,

              validation_data = (X_test, y_test))
# Evaluate the model's performance on the test data

score = CNN_model.evaluate(X_test, y_test, verbose=1)



print('Test loss:', score[0])

print('Test accuracy:', score[1])
# import pickle

# from pickle import Pickler

# CNN_model.dump('CNN_model.pkl')