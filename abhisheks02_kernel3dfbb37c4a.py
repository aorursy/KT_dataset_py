# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print(os.listdir('../input'))
#IMPORTING LIBRARIES

import glob

import cv2

from pathlib import Path

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

from keras.models import Sequential
directory = Path('../input/cell-images-for-detecting-malaria/cell_images/cell_images')



Parasitized_dir = directory / 'Parasitized'

Uninfected_dir  = directory / 'Uninfected'
X = []

Y = []
# DATA CLEANSING

Parasitized  = Parasitized_dir.glob('*.png')

Uninfected   = Uninfected_dir.glob('*.png')
height = 64

width  = 64

for u in Uninfected:

    image = cv2.imread(str(u))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(0, num_classes=2)

    X.append((img))

    Y.append((label))

    

for p in Parasitized:

    image = cv2.imread(str(p))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(1, num_classes=2)

    X.append((img))

    Y.append((label))

    

X = np.array(X)

Y = np.array(Y)
X.shape , Y.shape
X , Y = shuffle(X , Y , random_state = 42)
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)
model = Sequential()



model.add(Conv2D(16,(3,3),   padding='same' ,activation="relu" ,input_shape=(64,64,3)))

model.add(Conv2D(16,(3,3),  padding='same' ,activation="relu"))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(32,(3,3),  padding='same' ,activation="relu"))

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

          

model.add(Conv2D(32,(3,3), padding='same' ,activation="relu"))

model.add(BatchNormalization())

          

model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
datagen = ImageDataGenerator(

    zoom_range = 0.3 ,

    rotation_range=20,

    width_shift_range=0.2,

    horizontal_flip=True,

    fill_mode = 'nearest')





datagen.fit(X)




# fits the model on batches with real-time data augmentation:

model.fit_generator(datagen.flow(X_train , Y_train, batch_size=64),

                    steps_per_epoch=len(X) / 64, 

                    epochs=20,

                    validation_data=(X_test , Y_test))

loss, test_accuracy = model.evaluate(X_test, Y_test)

print(loss)

print(test_accuracy)