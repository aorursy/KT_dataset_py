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
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense,Conv2D, Flatten,Dropout, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.models import Sequential

from keras.callbacks import LearningRateScheduler
Train_Data = "../input/train.csv"
Base_dataset = np.loadtxt(Train_Data, skiprows=1, dtype='int', delimiter=',')

X_Train, X_Val, Y_Train, Y_Val = train_test_split(Base_dataset[:,1:], Base_dataset[:,0],test_size = 0.1)
X_Train = X_Train.reshape(-1, 28, 28, 1)

X_Val = X_Val.reshape(-1, 28, 28, 1)



X_Train = X_Train.astype("float32")/255.

X_Val = X_Val.astype("float32")/255.
Y_Train = to_categorical(Y_Train)

Y_Val = to_categorical(Y_Val)
print(Y_Train[0])
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size= (3,3),

                 activation='relu', input_shape = (28,28,1) ))

model.add(BatchNormalization())



model.add(Conv2D(filters = 16, kernel_size= (3,3),

                 activation='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size= (3,3),

                 activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 32, kernel_size= (3,3),

                 activation='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(zoom_range= 0.1,

                             height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
Hist = model.fit_generator(datagen.flow(X_Train, Y_Train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=20, #Increase this when not on Kaggle kernel

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(X_Val[:400,:], Y_Val[:400,:]), #For speed

                           callbacks=[annealer])
plt.plot(Hist.history['loss'], color='b')

plt.plot(Hist.history['val_loss'], color='r')

plt.show()

plt.plot(Hist.history['acc'], color='b')

plt.plot(Hist.history['val_acc'], color='r')

plt.show()
y_act = model.predict(X_Val)

y_pred = np.argmax(y_act, axis=1)

y_tru = np.argmax(Y_Val, axis=1)

Conf_Matrix = confusion_matrix(y_tru,y_pred)

Conf_Matrix