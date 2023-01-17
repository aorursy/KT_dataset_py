# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')

df
Images = []

directory = '/kaggle/input/super-ai-image-classification/train/train/images'

for file in df['id']:

    image = cv2.imread(directory+'/'+file) 

    image = cv2.resize(image,(224,224)) 

    Images.append(image)
Images = np.array(Images)

Images.shape
import matplotlib.pyplot as plt

plt.imshow(Images[1724], cmap='gray')

plt.show()
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(64,(3,3), padding="same", activation="relu", input_shape=(224,224,3)))

model.add(Conv2D(64,(3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(128, (3,3), padding="same", activation="relu"))

model.add(Conv2D(128, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(Conv2D(512, (3,3), padding="same", activation="relu"))

model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(4096,activation="relu"))

model.add(Dense(4096,activation="relu"))

model.add(Dense(2, activation="softmax"))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 

              loss=keras.losses.sparse_categorical_crossentropy)
model.summary()
datagen = keras.preprocessing.image.ImageDataGenerator()

batch = 6

datagen.fit(Images)

num_batch = len(Images) / batch

Progbar = keras.utils.Progbar(num_batch)

for epoch in range(50):

  batches = 0

  for x, y in datagen.flow(Images, df['category'], batch_size=batch):

    history = model.fit(x, y, verbose=0)

    batches += 1

    Progbar.update(batches, values=[('loss', history.history['loss'][0])])

    if batches >= num_batch:

      print(epoch)

      break
Xtest = []

ID = []

directory = '/kaggle/input/super-ai-image-classification/val/val/images'

for file in os.listdir(directory):

    image = cv2.imread(directory+'/'+file)

    image = cv2.resize(image,(224,224))

    Xtest.append(image)

    ID.append(file)

Xtest = np.array(Xtest)
Ztest = model.predict(Images[1500:])

np.sum(Ztest.argmax(axis=1) == df['category'][1500:])/len(Ztest)
Ztest = model.predict(Xtest)

Ztest = np.argmax(Ztest, axis=-1)

print(Ztest)
data2 = {'id':ID,'category':Ztest}

val = pd.DataFrame(data2)

val.to_csv("Submit.csv",index=False)