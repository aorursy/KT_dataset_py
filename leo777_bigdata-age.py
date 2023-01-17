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
def age_mask(age):

    y_age = np.zeros(len(age))

    flag = 1

    for i in range(len(age)):

        if age[i] == 1:

            flag = 0

            return y_age

        if flag == 1:

            y_age[i] = 1

            continue
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

y = []

for i in face_list:

    y.append(i[-6:-4])
y = np.array(y).astype(int)

y = y.flatten().tolist()

y = np.array(pd.get_dummies(y))

for i in range(len(y)):

    y[i] = age_mask(y[i])

X = face.reshape(-1, 64, 64, 1)

X = np.array(X / 255.0)
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

X_train, y_train = X, y
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.15,

    height_shift_range=0.15,

    shear_range=10,

    horizontal_flip=True)

datagen.fit(X_train)
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
model = Sequential()

model.add(Conv2D(20, kernel_size=(5,5), input_shape=X.shape[1:], activation='relu'))

model.add(Conv2D(40, kernel_size=(7,7), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(80, kernel_size=(9,9), activation='relu'))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(79, activation='sigmoid'))



model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),steps_per_epoch=len(X_train) / 64, epochs=95)
#train_loss = []

#test_loss = []
'''

for i in range(20):

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),steps_per_epoch=len(X_train) / 64, epochs=1)

    train_loss.extend(history.history['loss'])

    TR = model.evaluate(X_test, y_test)

    test_loss.append(TR[0])

'''
'''

df = pd.DataFrame(train_loss, columns=['train_loss'])

df['test_loss'] = test_loss

df.plot().set(xlabel='epoch', ylabel='mse')

'''
model.save('age_model.h5')