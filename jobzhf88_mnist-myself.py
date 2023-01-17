# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division, print_function, unicode_literals



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import tensorflow as tf



from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models



from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

import pandas as pd



model = Sequential()
X = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



y = X["label"]

X.drop(["label"], inplace = True, axis = 1)

print("Data are Ready!!")
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2 , random_state=42)
X_train.shape
X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')
X_train.shape
print("the number of training examples = %i" % X_train.shape[0])

print("the number of classes = %i" % len(np.unique(y_train)))

print("Dimention of images = {:d} x {:d}  ".format(X_train[1].shape[0],X_train[1].shape[1])  )



#This line will allow us to know the number of occurrences of each specific class in the data

unique, count= np.unique(y_train, return_counts=True)

print("The number of occuranc of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )

 

images_and_labels = list(zip(X_train,  y_train))

for index, (image, label) in enumerate(images_and_labels[:12]):

    plt.subplot(5, 4, index + 1)

    plt.axis('off')

    plt.imshow(image.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('label: %i' % label )
from keras.layers import Dropout



model.add(Conv2D(50, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))

model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Dropout(0.2))

model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))

model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Dropout(0.2))

model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))

model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Dropout(0.2))



model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Dropout(0.2))
from keras.layers.core import Activation



model.add(Flatten())

model.add(Dense(units=100, activation='relu'  ))

model.add(Dropout(0.3))



model.add(Dense(10))

model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train).astype('int32')

y_test = np_utils.to_categorical(y_test)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=40,epochs=50,validation_data = (X_test,y_test))
scores = model.evaluate(X_test, y_test, verbose = 10 )

print ( scores )
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

results = model.predict(test.values.reshape(-1,28,28,1))



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)
submission.head(10)