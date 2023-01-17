# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

print(tf.__version__)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test=pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
ids=test.id

test=test.drop(["id"],axis=1)

train_result=train.label

train=train.drop(["label"],axis=1)

test=test.iloc[:,:].values

test=test.reshape(-1,28,28,1)

y=train_result.iloc[:].values

x=train.iloc[:,:].values

x=x.reshape(-1,28,28,1)

x, test= x/255.0,test/255.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Considering to do data augumentation

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.25,  

        width_shift_range=0.25, 

        height_shift_range=0.25)
#The whole "building" of neuronetwork

Sequential=keras.Sequential

#Now we choose "who will be moved into" each layer

#possible Candidates are:

#Use different filters to get different channels.

Conv2D=keras.layers.Conv2D

#Pooling, select Maximum value from a certain pool.

MaxPool2D=keras.layers.MaxPool2D

#a flat ANN layer

Dense=keras.layers.Dense

#as it suggests, flatten: \mathbb{R}^2 \to \mathbb{R}

Flatten=keras.layers.Flatten

#Randomly select a chosen percentage of neurons and remove them temporarily

Dropout=keras.layers.Dropout

#BarchNormalization, "Centralize" the batch.(randomly chose a noise on sigma is available)

BatchNormalization=keras.layers.BatchNormalization
def generate_model():



    model=Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 

                 activation ='relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 

                 activation ='relu', input_shape = (28,28,1)))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 

                 activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(Dropout(0.3))

    model.add(Dense(10, activation = "softmax"))

    model.compile(keras.optimizers.Adam(0.0001), 

             loss="sparse_categorical_crossentropy",

              metrics=["accuracy"]

             )

    

    return model
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
#See how many epochs should be reasonable

model=generate_model()

# Fit the model

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=40),

                              epochs = 45, 

                              validation_data=(x_test,y_test),

                              validation_steps=int(x_test.shape[0]/40),

                              verbose = 1, steps_per_epoch=x_train.shape[0] / 40)
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predictions=model.predict_classes(test)

output=pd.DataFrame({"id":ids, "label":predictions})

output.to_csv("submission.csv",index=False)