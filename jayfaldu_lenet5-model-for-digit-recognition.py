# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing datasets



df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df2 = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print(df.shape)

print(df2.shape)
from sklearn.preprocessing import OneHotEncoder



x = np.array(df.drop('label',axis=1))/255

y = np.array(df.label)

test = np.array(df2)/255



enc = OneHotEncoder(sparse=False)

y= y.reshape((-1,1))

y = enc.fit_transform(y)



print(x.shape)

print(y.shape)

print(test.shape)
x_2d = x.reshape((x.shape[0],28,28,1))

test_2d = test.reshape((test.shape[0],28,28,1))



print(x_2d.shape)

print(test_2d.shape)
for i in range(5):

    plt.figure()

    plt.imshow(np.squeeze(x_2d[i]))
from sklearn.model_selection import train_test_split as tts



x_train,x_test,y_train,y_test = tts(x_2d,y,test_size = 0.15, random_state=42)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# importing required librery and moduls



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras import applications



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau
model = Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



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

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))





model.summary()

# setting for early stopping



from tensorflow import keras



callbacks = [

    keras.callbacks.EarlyStopping(

        # Stop training when `val_loss` is no longer improving

        monitor='val_loss',

        # "no longer improving" being defined as "no better than 1e-2 less"

        min_delta=1e-5,

        # "no longer improving" being further defined as "for at least 2 epochs"

        patience=15,

        verbose=1)

]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(x_2d,y,batch_size=64,epochs=600,validation_data=(x_test,y_test))



plt.plot(history.history['loss'],label='train_loss')

plt.plot(history.history['val_loss'],label='val_loss')

plt.xlabel('No. epoch')

plt.legend()

plt.show()



plt.figure()



plt.plot(history.history['accuracy'],label='train_accuracy')

plt.plot(history.history['val_accuracy'],label='test_accuracy')

plt.xlabel('No. epoch')

plt.legend()

plt.show()
scores = model.evaluate(x_test, y_test, verbose = 10 )

print ( scores )
predictions=model.predict(test_2d)

pre=predictions.argmax(axis=-1)
submission = pd.Series(pre,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)

submission.to_csv("final_submission_lenet5.csv",index=False)

submission.head()