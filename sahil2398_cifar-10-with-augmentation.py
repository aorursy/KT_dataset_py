# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))'''



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



# One hot encoding.

y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 256

x_test = x_test.astype('float32') / 256
def generate_model():

    return tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Conv2D(32, (3, 3)),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Dropout(0.25),



    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Conv2D(64, (3, 3)),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Dropout(0.25),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10),

    tf.keras.layers.Activation('softmax')

  ])



model = generate_model()
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)



model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.97):

            print("\nReached 97% accuracy so cancelling training!")

            self.model.stop_training = True 
datagen = ImageDataGenerator(

            zca_epsilon=1e-06,

            rotation_range=10,

            width_shift_range=0.1,

            height_shift_range=0.1,

            zoom_range=0.1,

            fill_mode='nearest',

            cval=0.,

            horizontal_flip=True

            )
datagen.fit(x_train)
callbacks = myCallback()



history = model.fit_generator(datagen.flow(x_train, y_train,

                                    batch_size=32),

                                    epochs=40,

                                    validation_data=(x_test, y_test),

                                    callbacks=[callbacks],

                                    workers=4)
import matplotlib.pyplot as plt



def plotmodelhistory(history): 

    fig, axs = plt.subplots(1,2,figsize=(15,5)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy']) 

    axs[0].plot(history.history['val_accuracy']) 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss']) 

    axs[1].plot(history.history['val_loss']) 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper left')

    plt.show()



# list all data in history

print(history.history.keys())



plotmodelhistory(history)
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])



# make prediction.

pred = model.predict(x_test)
pred
pred.shape
results = np.argmax(pred,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,10001),name = "id"),results],axis = 1)



print(submission)



submission.to_csv("submission.csv",index=False)