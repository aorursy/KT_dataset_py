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
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import tensorflow as tf

import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten



from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.shape
y_train = train["label"]

x_train = train.drop(labels = ["label"], axis=1)
# normalisation

x_train = x_train/255

test = test/255
# convert to tensor

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1, 28, 28, 1)
#y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=10)
y_train[0]
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=5)
plt.imshow(x_tr[0][:,:,0])
model = tf.keras.models.Sequential([

    Conv2D(filters = 32, kernel_size = (5,5), activation="relu", padding = "Same", input_shape=(28,28,1)),

    MaxPool2D(pool_size=(2,2)),

    

    Conv2D(filters = 32, kernel_size = (4,4), activation="relu", padding = "Same"),

    MaxPool2D(pool_size=(2,2)),

    Dropout(0.25),

    

    Conv2D(filters=64, kernel_size = (3,3), activation="relu", padding = "Same"),

    MaxPool2D(pool_size=(2,2)),

    

    Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding = "Same"),

    MaxPool2D(pool_size=(2,2)),

    Dropout(0.25),

    

    Flatten(),

    

    Dense(256, activation="relu"),

    Dropout(0.25),

    

    Dense(10, activation = "softmax")

])
model.compile(optimizer="adam",

              loss=tf.keras.losses.sparse_categorical_crossentropy,

              metrics=["accuracy"])
EPOCHS = 15

BATCH_SIZE = 32
#history = model.fit(x_tr, y_tr,

#          batch_size=BATCH_SIZE,

#          epochs=EPOCHS,

#          verbose=1,

#          validation_data=(x_val, y_val))
# 5 Epochs scored: 0.98557 on public LB
def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs_range = range(EPOCHS)



    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')

    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')

    plt.title('Training and Validation Accuracy')



    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.title('Training and Validation Loss')

    plt.show()
datagen = ImageDataGenerator(rotation_range=20,

                              width_shift_range=0.2,

                              height_shift_range=0.2,

                              zoom_range=0.2)
datagen.fit(x_tr)
history = model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = BATCH_SIZE),

                   steps_per_epoch = len(x_tr) / BATCH_SIZE,

                   epochs = EPOCHS,

                   validation_data = (x_val, y_val))
plot_history(history)
model.summary()
model = tf.keras.models.Sequential([

    Conv2D(filters = 32, kernel_size = (5,5), activation="relu", padding = "Same", input_shape=(28,28,1)),

    Conv2D(filters = 32, kernel_size = (4,4), activation="relu", padding = "Same"),

    MaxPool2D(pool_size=(2,2)),

    Dropout(0.25),

    

    Conv2D(filters=64, kernel_size = (3,3), activation="relu", padding = "Same"),    

    Conv2D(filters = 64, kernel_size = (2,2), activation="relu", padding = "Same"),

    MaxPool2D(pool_size=(2,2)),

    Dropout(0.25),

    

    Flatten(),

    

    Dense(256, activation="relu"),

    Dropout(0.25),

    

    Dense(10, activation = "softmax")

])
model.summary()
model.compile(optimizer="adam",

             loss = tf.keras.losses.sparse_categorical_crossentropy,

             metrics=["accuracy"])
history = model.fit_generator(datagen.flow(x_tr, y_tr, batch_size=BATCH_SIZE),

                             epochs = EPOCHS,

                             validation_data = (x_val, y_val))
plot_history(history)
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)