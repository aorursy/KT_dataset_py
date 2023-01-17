from __future__ import print_function



import pandas as pd

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import RMSprop

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

#%matplotlib inline
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
# Download the data

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
y_train = train["label"].to_numpy()

x_train = train.drop(columns = ["label"]).to_numpy()
x_test = test.to_numpy()
# Let's preview the labels

preview = sns.countplot(y_train)
# use one-hot enconding

y_train = keras.utils.to_categorical(y_train, 10)



print("The cateragories of these image are now encoded as : \n",y_train[0][:],"\n",y_train[1][:],"\n",y_train[2][:],"\n",y_train[3][:],"\n" )
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255



x_train.shape
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))





model.summary()



model.compile(loss="categorical_crossentropy",

              optimizer=keras.optimizers.adam(),

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=256,

          epochs=15,

          verbose=1)
# Predict classes for the test set

pred = model.predict(x_test)



# select the indix with the maximum probability

pred = np.argmax(pred,axis = 1)
# Submit

submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),

                         "Label": pred})

submissions.to_csv("submission.csv", index=False, header=True)