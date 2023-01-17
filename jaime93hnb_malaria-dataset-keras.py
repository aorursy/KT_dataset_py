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
import cv2

import keras

from PIL import Image

from keras.utils import to_categorical, np_utils

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization



import matplotlib.pyplot as plt



import keras.backend as K
images = []

labels = []





path_0 = "../input/cell_images/cell_images/Parasitized/"

path_1 = "../input/cell_images/cell_images/Uninfected/"



paras = os.listdir(path_0)

uninf = os.listdir(path_1)



for p, u in zip(paras, uninf):

    try:

        img1 = cv2.imread(path_0 + p)



        img_to_array = Image.fromarray(img1, 'RGB')

        scale_img = img_to_array.resize((64,64))

        images.append(np.array(scale_img))

        labels.append(0)



        img2 = cv2.imread(path_1 + u)



        img_to_array = Image.fromarray(img2, 'RGB')

        scale_img = img_to_array.resize((64,64))

        images.append(np.array(scale_img))

        labels.append(1)

        

    except AttributeError:

        print(" ")
len(images)
plt.imshow(img1)
plt.imshow(img2)
X = np.array(images)

y = np.array(labels)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test = X_test.astype("float32") / 255

X_train = X_train.astype("float32") / 255
y_train = to_categorical(y_train, 2)

y_test = to_categorical(y_test, 2)
K.clear_session()



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(64,64,3)))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(128, activation='relu'))



model.add(Dropout(0.2))

model.add(Dense((64)))



model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))



model.summary()
adam =  keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=80, epochs=30,verbose=1)
history = model.history

history.history.keys()
plt.plot(history.epoch, history.history['acc'], 'b', label='Accuracy')

plt.plot(history.epoch, history.history['loss'], color='orange', label='Loss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('%')

plt.title('Model Accuracy')

plt.show()
accuracy = model.evaluate(X_test, y_test, verbose=1)



print('\n', 'Test Accuracy:-', accuracy[1])