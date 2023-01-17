from keras import models

from keras import layers

from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout,ZeroPadding2D,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import numpy as np

from matplotlib import pyplot

import os

import zipfile

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop,Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random



Train = pd.read_csv("../input/train.csv")

Test = pd.read_csv("../input/test.csv")
Train.head()
#Read the labels into a seprate array

y_train = Train['label'].values



#convert the images into a matrix of size 42,000 X 28 X 28 

train_images = Train.loc[:,Train.columns != 'label'].values

train_images = train_images.reshape(-1,28,28,1)

train_images.shape



#convert the images into a matrix of size 28,000 X 28 X 28 

test_images = Test.values

test_images_final = test_images.reshape(-1,28,28,1)

test_images_final.shape

# create a grid of 3x3 images

for i in range(0, 9):

    pyplot.subplot(330 + 1 + i)

    pyplot.imshow(train_images[i].reshape(28,28), cmap=pyplot.get_cmap('gray'))

# show the plot

pyplot.show()
#reshape data to fit model



train_images = train_images.astype('float32') /255



test_images_final = test_images_final.astype('float32') /255
import tensorflow as tf

network = tf.keras.models.Sequential([

    tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None,input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (5,5), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None),

    tf.keras.layers.Conv2D(64, (6,6), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None),

    tf.keras.layers.Conv2D(64, (7,7), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(5,5),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.45),

    tf.keras.layers.Dense(10, activation='softmax')

])
from keras import optimizers



#network.compile(optimizer=RMSprop(lr=0.001),

network.compile(optimizer=Adam(lr=0.001),

loss='categorical_crossentropy',

metrics=['accuracy'])
network.summary()
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_images, y_train, test_size=0.15)
from keras.utils import to_categorical

train_labels = to_categorical(y_train)

test_labels = to_categorical(y_test)

y_train = train_labels

y_train.shape



#datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

datagen = ImageDataGenerator( rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

# fit parameters from data

datagen.fit(X_train)



np.concatenate((X_train,X_train),axis=0)

random.seed(12345)

for X_batch, y_batch in datagen.flow(np.concatenate((X_train,X_train),axis=0), np.concatenate((y_train,y_train),axis=0), batch_size=35700):

  break

 

  

X_trainE = X_batch

y_trainE = y_batch





for i in range(0, 9):

        pyplot.subplot(330 + 1 + i)

        pyplot.imshow(X_trainE[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

# show the plot

pyplot.show()







X_Combine = np.concatenate((X_train,X_trainE),axis=0)

y_combine = np.concatenate((y_train,y_trainE),axis=0)

print(X_Combine.shape)

print(y_combine.shape)
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('val_acc')>=0.9930):

      print("\nReached 99.0% accuracy so cancelling training!")

      self.model.stop_training = True

      

callbacks = myCallback()
history = network.fit(X_Combine, y_combine, epochs=60, batch_size=512,validation_data=(X_test,test_labels),callbacks=[callbacks])
test_loss, test_acc = network.evaluate(X_test, test_labels)

print('test_acc:', test_acc)
#Make Prediction

predict = network.predict(test_images_final)

#Convert the results to the digits value 

y_classes = [np.argmax(y, axis=None, out=None) for y in predict]
import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range (1,len(history_dict['loss'])+1)

#epochs = range(1, 36)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
x = list(range(1, 28001))

df = pd.DataFrame({'ImageId' :x,'Label':y_classes})

df.to_csv("output.csv",index=False)
