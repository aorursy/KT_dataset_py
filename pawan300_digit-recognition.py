# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directoryka

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.shape, test.shape
# Normalize the data

train = train / 255.0

test = test / 255.0
y = train["label"]

y = tf.keras.utils.to_categorical(y, num_classes=10)

image_id = list(test.index)

image_id = [i+1 for i in image_id]



train = train.drop("label", axis=1)

train = train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
train.shape
def plot_image(index):

    plt.figure(0, figsize=(4,4))

    plt.imshow(np.reshape(train[index],(28,28)))
plot_image(300)
def plot_mulitple_images(columns, lines, index):

    plt.figure(0, figsize=(10,10))

    for i in range(columns*lines):

        plt.subplot(lines, columns, i+1)

        plt.imshow(np.reshape(train[index+i], (28,28)))

        plt.axis('off')
plot_mulitple_images(5, 6, 34)
import keras, os

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPool2D, BatchNormalization, MaxPool2D

from keras.callbacks import ModelCheckpoint, EarlyStopping
xtrain, xtest, ytrain, ytest = train_test_split(train, y, test_size=0.2)

xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2)
xtrain.shape, xval.shape, xtest.shape
from matplotlib import pyplot as plt

import math

from keras.callbacks import LambdaCallback

import keras.backend as K





class LRFinder:

    """

    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.

    See for details:

    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

    """

    def __init__(self, model):

        self.model = model

        self.losses = []

        self.lrs = []

        self.best_loss = 1e9



    def on_batch_end(self, batch, logs):

        # Log the learning rate

        lr = K.get_value(self.model.optimizer.lr)

        self.lrs.append(lr)



        # Log the loss

        loss = logs['loss']

        self.losses.append(loss)



        # Check whether the loss got too large or NaN

        if math.isnan(loss) or loss > self.best_loss * 4:

            self.model.stop_training = True

            return



        if loss < self.best_loss:

            self.best_loss = loss



        # Increase the learning rate for the next batch

        lr *= self.lr_mult

        K.set_value(self.model.optimizer.lr, lr)



    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):

        num_batches = epochs * x_train.shape[0] / batch_size

        self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)



        # Save weights into a file

        self.model.save_weights('tmp.h5')



        # Remember the original learning rate

        original_lr = K.get_value(self.model.optimizer.lr)



        # Set the initial learning rate

        K.set_value(self.model.optimizer.lr, start_lr)



        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))



        self.model.fit(x_train, y_train,

                        batch_size=batch_size, epochs=epochs,

                        callbacks=[callback])



        # Restore the weights to the state before model fitting

        self.model.load_weights('tmp.h5')



        # Restore the original learning rate

        K.set_value(self.model.optimizer.lr, original_lr)



    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):

        """

        Plots the loss.

        Parameters:

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

        """

        plt.ylabel("loss")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])

        plt.xscale('log')



    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):

        """

        Plots rate of change of the loss function.

        Parameters:

            sma - number of batches for simple moving average to smooth out the curve.

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

            y_lim - limits for the y axis.

        """

        assert sma >= 1

        derivatives = [0] * sma

        for i in range(sma, len(self.lrs)):

            derivative = (self.losses[i] - self.losses[i - sma]) / sma

            derivatives.append(derivative)



        plt.ylabel("rate of loss change")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])

        plt.xscale('log')

        plt.ylim(y_lim)
def model():



    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))



    adam = tf.keras.optimizers.Adam()



    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])



    early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)



    checkpoint_path = 'training_1/cp.ckpt'

    checkpoint_dir = os.path.dirname(checkpoint_path)



    # create checkpoint callback

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                save_weights_only=True,

                                                verbose=1)

    return model


def determineLearningRate(xtrain,ytrain,xtest,ytest):    

    input_shape = (28,28,1)

    num_classes = 10

    epochs = 15

    

    lr_finder = LRFinder(model())

    lr_finder.find(xtrain,ytrain, start_lr=0.000001, end_lr=100, batch_size=64, epochs=epochs)

    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)

    plt.show()

    return model

determineLearningRate(xtrain, ytrain, xtest, ytest)
def model():



    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))



    adam = tf.keras.optimizers.Adam(

    learning_rate=0.001,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-07,

    amsgrad=False)



    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])



    early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)



    checkpoint_path = 'training_1/cp.ckpt'

    checkpoint_dir = os.path.dirname(checkpoint_path)



    # create checkpoint callback

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                save_weights_only=True,

                                                verbose=1)

    print(model.summary())

    history = model.fit(xtrain, ytrain, epochs=100, callbacks=[cp_callback, early], validation_data=(xval, yval))

    prediction = model.predict(test)

    prediction = np.argmax(prediction, axis=1)



    return history, prediction
history, prediction = model()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
data = {"ImageId": image_id, "Label":prediction}

results = pd.DataFrame(data)

results.to_csv("result.csv",index=False)
def plot_mulitple_images_with_label(columns, lines, index):

    plt.figure(0, figsize=(12,12))

    for i in range(columns*lines):

        plt.subplot(lines, columns, i+1)

        plt.imshow(np.reshape(test[index+i], (28,28)))

        plt.title("Predicted : "+str(prediction[index+i]))

        plt.axis('off')
plot_mulitple_images_with_label(5, 3, 27)
from keras.preprocessing.image import ImageDataGenerator
datagen_train = datagen_valid = ImageDataGenerator(

        featurewise_center = False,

        samplewise_center = False,

        featurewise_std_normalization = False, 

        samplewise_std_normalization = False,

        zca_whitening = False,

        horizontal_flip = False,

        vertical_flip = False,

        fill_mode = 'nearest',

        rotation_range = 10,  

        zoom_range = 0.1, 

        width_shift_range = 0.1, 

        height_shift_range = 0.1)

        



datagen_train.fit(xtrain)

train_gen = datagen_train.flow(xtrain, ytrain, batch_size=64)

datagen_valid.fit(xval)

valid_gen = datagen_valid.flow(xval, yval, batch_size=64)
model = Sequential([

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape = (28,28,1)),

        Conv2D(32, kernel_size=(3, 3), activation='relu' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        Conv2D(64, kernel_size=(3, 3), activation='relu' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),

        Conv2D(128, kernel_size=(3, 3), activation='relu' ),

        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),

        Dropout(0.2),

        

        

        Flatten(),

        

        Dense(512, activation='relu'),

        Dropout(0.5),

        

        Dense(10, activation = "softmax")

        

    ])

adam = tf.keras.optimizers.Adam(

    learning_rate=0.001,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-07,

    amsgrad=False)



model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)



checkpoint_path = 'training_1/cp.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)



cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                save_weights_only=True,

                                                verbose=1)

print(model.summary())

history = model.fit((train_gen), epochs=100, callbacks=[cp_callback, early], validation_data=(valid_gen))

prediction = model.predict(test)

prediction = np.argmax(prediction, axis=1)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
data = {"ImageId": image_id, "Label":prediction}

results = pd.DataFrame(data)

results.to_csv("result_data_generator.csv",index=False)
plot_mulitple_images_with_label(5, 3, 27)