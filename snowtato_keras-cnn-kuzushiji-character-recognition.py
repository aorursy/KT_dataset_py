import numpy as np

import os

import matplotlib.pyplot as plt

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten

from sklearn.preprocessing import MinMaxScaler
Xtrain = np.load("../input/k49-train-imgs.npz")['arr_0']

ytrain = np.load("../input/k49-train-labels.npz")['arr_0']

Xtest = np.load("../input/k49-test-imgs.npz")['arr_0']

ytest = np.load("../input/k49-test-labels.npz")['arr_0']





train_one_hot_labels = keras.utils.to_categorical(ytrain, num_classes=49)

test_one_hot_labels = keras.utils.to_categorical(ytest, num_classes=49)

n_train = ytrain.shape[0]

n_test = ytest.shape[0]

npix = Xtrain.shape[1]



Xtrain1 = Xtrain.reshape(n_train, -1)

Xtest1 = Xtest.reshape(n_test, -1)

scaler = MinMaxScaler()

Xtrain1 = scaler.fit_transform(Xtrain1).astype('float32')

Xtest1 = scaler.fit_transform(Xtest1).astype('float32')
temp = Xtrain.reshape(n_train, -1)

np.sum(np.min(temp, axis=0)), np.sum(np.max(temp, axis=0)/255)
model = Sequential([

    Dense(128, input_shape=(784,)),

    Activation('relu'),

    Dropout(rate=0.5),

    Dense(49),

    Activation('softmax'),

])



# For a multi-class classification problem

model.compile(optimizer='adagrad',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit(Xtrain1, train_one_hot_labels, epochs=100, batch_size=256, validation_data = (Xtest1, test_one_hot_labels))
def plot_history(history):

    # Plot training & validation accuracy values

    plt.figure(figsize=(12,5))

    plt.subplot(121)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

#     plt.show()



    # Plot training & validation loss values

    plt.subplot(122)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
plot_history(history)
model2 = Sequential([

    Dense(64, input_shape=(784,)),

    Activation('relu'),

    Dropout(rate=0.2),

    Dense(64),

    Activation('relu'),

    Dropout(rate=0.2),

    Dense(49),

    Activation('softmax'),

])



# For a multi-class classification problem

model2.compile(optimizer='adagrad',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
history2 = model2.fit(Xtrain1, train_one_hot_labels, epochs=100, batch_size=256, validation_data = (Xtest1, test_one_hot_labels))
plot_history(history2)
from keras import backend as K

K.image_data_format()





if K.image_data_format() == 'channels_last':

    Xtrain2d = Xtrain.reshape(n_train, npix, npix, 1).astype('float32')/255

    Xtest2d = Xtest.reshape(n_test, npix, npix, 1).astype('float32')/255

    input_shape = (npix, npix, 1)

else:

    print("Images not resized")
model3 = Sequential()

model3.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model3.add(Conv2D(64, (3, 3), activation='relu'))

model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Dropout(0.25))

model3.add(Flatten())

model3.add(Dense(128, activation='relu'))

model3.add(Dropout(0.5))

model3.add(Dense(49, activation='softmax'))

model3.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
history3 = model3.fit(Xtrain2d, train_one_hot_labels, epochs=100, batch_size=128, validation_data = (Xtest2d, test_one_hot_labels))
plot_history(history3)