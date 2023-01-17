# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
validation = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
validation.head(5)
y_train = train['label']

x_train = train.drop(labels = ['label'], axis = 1 )
print(y_train.shape)

y_train.head(5)
print("NULL Value Check for Train X:\n",x_train.isnull().sum())

print("NULL Value Check for Train Y: ",y_train.isnull().sum())

print("NULL Value Check for Testset:\n",validation.isnull().sum())
countplot = y_train.value_counts().plot(kind = 'bar',figsize=(5,3),fontsize=13,color = 'red')

plt.show()
countplot = sns.countplot(y_train)
nrows,ncols = 10,10

plt.figure(figsize=(10,10))

for digits in range(0,0):

    plt.subplot(nrows,ncols,digits+1,frameon=False)

    next_digit = x_train.iloc[digits].as_matrix().reshape(28,28)

    plt.imshow(next_digit,cmap = "gray")

plt.show()
x_train = x_train/ 255.0

validation = validation / 255.0

x_train = x_train.values.reshape(-1,28,28,1)

validation = validation.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train, num_classes = 10)
print("Shape of X train: ",x_train.shape)

print("Shape of Y train: ",y_train.shape)

print("Shape of ValidationSet:",validation.shape)
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,random_state = 42,test_size= 0.15)
print("X_train shape: {}, X_test: {}".format(x_train.shape,x_test.shape))

print("Y_train shape: {}, Y_test: {}".format(y_train.shape,y_test.shape))
def digit_pred_model():  

    accuracy_limit = 0.996

    class vCallback(tf.keras.callbacks.Callback):

        def epoch_limit(self,epoch,logs={}):

            if(logs.get('acc')>accuracy_limit):

                print("\nAccuracy %99.6, cancelling training.")

                self.model.stop_training = True

    callbacks = vCallback()    

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32,(5,5),padding = 'Same',activation = 'relu', input_shape = (28,28,1)),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3,3),padding = 'Same', activation='relu'),

        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(20,activation = 'relu'),

        tf.keras.layers.Dense(10, activation='softmax'),

        ])

    

    from tensorflow.keras.optimizers import RMSprop,Adam

    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['acc'])   

    history = model.fit(x_train,y_train,epochs = 30,validation_data = (x_test,y_test), verbose = 2, batch_size=32 ,callbacks = [callbacks])

    

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



    return history.history['acc'][-1]
digit_pred_model()