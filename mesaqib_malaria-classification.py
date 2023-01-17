# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os,shutil

#from tqdm import tqdm

import random



%matplotlib inline





# Any results you write to the current directory are saved as output.
from keras import layers

from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

input_shape=(96,96, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(512, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

import random

import cv2

from PIL import Image
data = []

labels = []



parasitized = os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")

uninfected = os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")



parasitized.remove("Thumbs.db")

uninfected.remove("Thumbs.db")



for u in uninfected:

    image = Image.open("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/" + u)

    image = image.resize((96,96))

    image = img_to_array(image)

    data.append(image)

    labels.append(0)



for p in parasitized:

    image = Image.open("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/" + p)

    image = image.resize((96,96))

    image = img_to_array(image)

    data.append(image)

    labels.append(1)

data = np.array(data)

labels = np.array(labels)
n = np.arange(data.shape[0])

np.random.shuffle(n)

data = data[n]

labels = labels[n]
data = data.astype(np.float32)

#labels = labels.astype(np.int32)

data = data/255.0
train_x, x, train_y, y = train_test_split(data, labels, test_size = 0.2, random_state = 42)



val_x, test_x, val_y , test_y = train_test_split(x, y , test_size = 0.5, random_state = 42)
print("Number of trainig examples:",train_x.shape[0])

print("Number of validation examples:",val_x.shape[0])

print("Number of test examples:",test_x.shape[0])
aug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,

                        zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
from keras.callbacks import EarlyStopping

from keras.callbacks import ReduceLROnPlateau

callbacks_list = [ EarlyStopping(monitor='val_accuracy', patience = 20, min_delta = 1,verbose = 1),

                   ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=15,verbose = 1)]
opt = Adam(lr = 1e-3)

model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
history = model.fit_generator(aug.flow(train_x,train_y, batch_size = 32) ,steps_per_epoch = len(train_x)//32 ,

                              validation_data = (val_x,val_y), callbacks = callbacks_list, epochs = 60  )
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='lower right')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
_, test_acc = model.evaluate(test_x, test_y, verbose=0)

print('Test Accuracy: %.3f' % (test_acc))
model.save('my_model.h5')