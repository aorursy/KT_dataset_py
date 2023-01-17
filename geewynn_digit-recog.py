# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#shape of train and test

print(train.shape)

print(test.shape)
# Extract the target feature from the train data



target = train.label

train = train.drop(train[['label']], axis=1)
# Split the train data into train and test



X_train, X_test, y_train, y_test = train_test_split(train, target, test_size= 0.2)
# Convert into numpy arrays



X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)
#Reshape data



X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)
#Scale the data



X_train = X_train.astype(float) / 255.0

X_test = X_test.astype(float) / 255.0
def cat(df):

    df = to_categorical(df)

    return df

    

y_train = cat(y_train)

y_test = cat(y_test)
# Carrying out same data preprocessing on test data



test = np.array(test)

test = test.reshape(28000, 28,28, 1)

test = test.astype(float)/255.0
model =  Sequential()

model.add(Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

          

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))



model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(optimizer='adam', 

              loss='categorical_crossentropy',

              metrics=['accuracy']

              )
history = model.fit(X_train, y_train,

                   validation_data=(X_test, y_test),

                    epochs = 30,

                    batch_size= 42)
plt.plot(history.history['loss'], color='b')

plt.plot(history.history['val_loss'], color='r')

plt.show()

plt.plot(history.history['acc'], color='b')

plt.plot(history.history['val_acc'], color='r')

plt.show()
pred = model.predict(test)
finalpred = np.argmax(pred,axis=1)
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
subid = sub.ImageId

df = pd.DataFrame({'ImageId':subid, 'Label':finalpred})
df.to_csv('digits.csv')
# For the data augmentation we rotated the image by 30 degrees, adjusted the width and height of the images by 0.15

# Did a horizontal flip of images and lastly zoomed the image by 50%



datagen = ImageDataGenerator(

                    rotation_range=30, 

                    width_shift_range=.15, 

                    height_shift_range=.15, 

                    horizontal_flip=True, 

                    zoom_range=0.5

                    )
epoch = 30

batch_size= 42





history1 = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                           steps_per_epoch=500,

                           epochs=epoch,

                           verbose=2,

                           validation_data=(X_test, y_test),

                           )
pred = model.predict(test)
finalpred = np.argmax(pred,axis=1)
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
subid = sub.ImageId

df = pd.DataFrame({'ImageId':subid, 'Label':finalpred})
df.to_csv('digits1.csv')