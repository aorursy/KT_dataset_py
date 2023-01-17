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
gender = pd.read_csv('../input/face-gender/gender.csv', index_col=0)
import cv2 as cv



base = "../input/faceproj/resized_faces/resized_faces/"

a = []

for i in gender.index:

    img = cv.imread(base+i, 0)

    a.append(img.tolist())

X = np.array(a).reshape(-1, 64, 64, 1)

X = np.array(X / 255)

del a
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

y = np.array(gender).flatten()

X, y = shuffle(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

X_train, y_train = X, y
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=10, 

    horizontal_flip=True)

datagen.fit(X_train)
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

from keras.models import Sequential
model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=X.shape[1:]))

model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))

model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))

model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), steps_per_epoch=len(X_train) / 64, epochs=30)
'''

train_loss = []

test_loss = []

train_accu = []

test_accu = []

'''
'''

for i in range(30):

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), steps_per_epoch=len(X_train) / 64, epochs=1)

    train_loss.extend(history.history['loss'])

    train_accu.extend(history.history['acc'])

    d = model.evaluate(X_test, y_test)

    test_loss.append(d[0])

    test_accu.append(d[1])

'''
'''

df = pd.DataFrame({'train_loss':train_loss, 'test_loss':test_loss})

df.plot().set(xlabel='epoch', ylabel='binary_crossentropy')

df = pd.DataFrame({'train_acc':train_accu, 'test_acc':test_accu})

df.plot().set(xlabel='epoch', ylabel='accuracy')

'''
model.save('gender_model.h5')