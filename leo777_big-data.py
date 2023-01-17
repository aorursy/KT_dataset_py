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
import cv2 as cv



base = "../input/resized_faces/resized_faces/"

face_list = os.listdir(base)

a = []

for i in face_list:

    img = cv.imread(base+i, 0)

    temp = img.tolist()

    a.append(temp)

face = np.array(a)

del a
base = "../input/valid_1/valid_1/"

nface_list = os.listdir(base)

a = []

for i in nface_list:

    img = cv.imread(base+i, 0)

    temp = img.tolist()

    a.append(temp)

nonface = np.array(a)

del a
data = np.concatenate((face, nonface), axis=0)

y = np.zeros(len(face)+len(nonface))

for i in range(len(face)):

    y[i] = 1
from sklearn.utils import shuffle

X, y = shuffle(data, y)
X = X.reshape(-1, 64, 64, 1)

X = X / 255.0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=10)

datagen.fit(X)
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
model = Sequential()

model.add(Conv2D(20, kernel_size=(5,5), input_shape=X.shape[1:], activation='relu'))

model.add(Conv2D(40, kernel_size=(7,7), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(80, kernel_size=(9,9), activation='relu'))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), steps_per_epoch=len(X) / 64, epochs=25)
'''

train_loss = []

test_loss = []

train_accu = []

test_accu = []

'''
'''

for i in range(10):

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), steps_per_epoch=len(X) / 64, epochs=1)

    train_loss.extend(history.history['loss'])

    train_accu.extend(history.history['acc'])

    d = model.evaluate(X_test, y_test)

    test_loss.append(d[0])

    test_accu.append(d[1])

'''
'''

df = pd.DataFrame({'train_loss':train_loss, 'test_loss':test_loss})

df.plot().set(xlabel='epoch', ylabel='binary_crossentropy')

df = pd.DataFrame({'train_accu':train_accu, 'test_accu':test_accu})

df.plot().set(xlabel='epoch', ylabel='accuracy')

'''
model.save('face_model.h5')