# Importing Libraries

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils
# Retrieving training and test data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train[0].argmax(axis=0)
print('X_train shape: ', X_train.shape)

print('Y_train shape: ', y_train.shape)

print('X_test shape: ', X_test.shape)

print('Y_test shape: ', y_test.shape)
# Visualizing the training data



# displaying a train image by it's index

def display_image(index):

    image = X_train[index]

    label = y_train[index]

    plt.title("Training data, index: %d, Label: %d"%(index, label))

    plt.imshow(image, cmap='gray_r')

    plt.show()

    

display_image(2)
# Flattening the data

# (28, 28) - 2D; (784) - 1D

X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

# Rescaling

X_train /= 255

X_test /= 255



print("Training matrix shape: ", X_train.shape)

print("Testing matrix shape: ", X_test.shape)
# One Hot Encoding of labels

print("Shape of y_train before encoding: ", y_train.shape)

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)

print("Shape of y_train after One Hot encoding: ", y_train.shape)
# Building the model

def build_model():

    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(512))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model



model = build_model()
# Compiling the model

model.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

             metrics=['accuracy'])



# Training the model

model.fit(X_train, y_train,

         batch_size=128,

         epochs=5,

         validation_data=(X_test, y_test),

         verbose=1)
# Comparing the predicted and actual results

score = model.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)

print('Test Score: ', score[0])

print("Test Accuracy :", score[1])