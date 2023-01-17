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
# Reading the data

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

x_train = train.iloc[:,1:]

y_train = train.iloc[:,:1]
# Normalizing the pixels values

x_train = x_train/255.0

test = test/255.0



# Bringing dimensions in Keras expected form (examples, hx, hy, channels)

x_train = x_train.to_numpy().reshape(x_train.shape[0],28,28,1)

test = test.to_numpy().reshape(test.shape[0],28,28,1)



print("Train Data Shape is ", x_train.shape)

print("Train Label data Shape is ", y_train.shape)

print("Test Data Shape is ", test.shape)
import tensorflow as tf

from tensorflow import keras



model = keras.Sequential([

    keras.layers.Conv2D(input_shape =(28,28,1), filters=16, kernel_size=(3,3), activation = 'relu',

                        strides=(1,1), data_format = "channels_last",),

    

    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu',

                        strides=(1,1), data_format = "channels_last"),

    keras.layers.MaxPool2D(2,2),

    

    keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding = 'same', activation = 'relu',

                        strides=(1,1), data_format = "channels_last"),

    keras.layers.Dropout(0.2),

    

    keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding = 'same', activation = 'relu',

                        strides=(1,1), data_format = "channels_last"),

    keras.layers.MaxPool2D(2,2),

    keras.layers.Dropout(0.3),

    

    keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation = 'relu',

                        strides=(1,1), data_format = "channels_last"),

    keras.layers.Dropout(0.3),

    keras.layers.BatchNormalization(),

    

#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), 

#                         strides=(1,1), data_format = "channels_last"),

#     keras.layers.BatchNormalization(),

    

    keras.layers.Flatten(),

    

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(10, activation="softmax")

])
from tensorflow.keras.optimizers import Adam

adm = Adam(learning_rate = 0.002)

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(

    rotation_range=20,

    zoom_range = 0.1,  

    width_shift_range=0.1, 

    height_shift_range=0.1

)



data_gen.fit(x_train)



len(x_train)
from keras.callbacks import ReduceLROnPlateau

reduce_LR = ReduceLROnPlateau(monitor='acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00005)
history = model.fit_generator(data_gen.flow(x_train,y_train, batch_size=64),

                    epochs = 50, 

                    verbose = 1,

                    callbacks = [reduce_LR])

# model.fit(x_train, y_train, epochs=10)
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predictions = model.predict(test)



# select the indix with the maximum probability

predictions = np.argmax(predictions,axis = 1)



predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), predictions],axis = 1)



submission.to_csv("cnn_mnist.csv",index=False)