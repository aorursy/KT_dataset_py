# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#PARASITIZED

img = plt.imread("../input/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png")

plt.imshow(img)
#UNINFECTED

img = plt.imread("../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png")

plt.imshow(img)
train_images = []

label = []

count = 0

for x in os.listdir("../input/cell_images/cell_images/Parasitized/"):

    if count % 1000 == 0:

        print(count)

    try:

        img = plt.imread("../input/cell_images/cell_images/Parasitized/" + x)

        img = cv2.resize(img, (32, 32))

        train_images.append(img)

        label.append(1)

        count += 1

    except:

        pass

    

count = 0   

for x in os.listdir("../input/cell_images/cell_images/Uninfected/"):

    if count % 1000 == 0:

        print(count)

    try:

        img = plt.imread("../input/cell_images/cell_images/Uninfected/" + x)

        img = cv2.resize(img, (32,32))

        train_images.append(img)

        label.append(0)

        count += 1

    except:

        pass
train_images = np.array(train_images)

label = np.array(label)

print(train_images.shape)

print(label.shape)
from keras.utils import to_categorical

label = to_categorical(label)

print(label.shape)
from keras.applications import DenseNet121

from keras.layers import Flatten, Dropout, GlobalAveragePooling2D, Dense

from keras.models import Sequential
conv_base = DenseNet121(weights="imagenet", include_top = False, input_shape=(32,32,3))
conv_base.summary()
model = Sequential()

model.add(conv_base)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', 

                  metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint   



checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit(

    x=train_images,

    y=label,

    batch_size=64,

    epochs=10,

    callbacks=[checkpoint],

    validation_split=0.2,

    verbose=1

)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['train','validation'])

plt.show()