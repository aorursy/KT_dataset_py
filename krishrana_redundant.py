# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import os

import pandas as pd

import matplotlib

import cv2
DATA_PATH="/kaggle/input/covidlarge/train"

CATEGORIES=['Covid', 'NORMAL', 'PNEUMONIA']

img_size=150

training_data = []

def create_training_data():

	#iterating through different categories

    for category in CATEGORIES:

        path=os.path.join(DATA_PATH, category)

        print(path)

        class_num = CATEGORIES.index(category)

        print(class_num)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array,(img_size,img_size)) #resizeing the all the images to same size(70,70)

                training_data.append([new_array,class_num])

            except Exception as e:

                pass
create_training_data()

print(len(training_data))
import random

random.shuffle(training_data)
X=[]

y=[]



for features,label in training_data: #new_array is features and class_num is label

	X.append(features)

	y.append(label)





X=np.array(X).reshape(-1, img_size, img_size, 1) #transforming X into numpy array

print(X.shape)

print(len(y))
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
X=X/255

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

#model.add(Conv2D(32,(3,3)))

#model.add(Activation('relu'))

model.add(MaxPooling2D(2,2))







#model.add(Conv2D(64, (3,3)))

#model.add(Activation('relu'))

model.add(Conv2D(64, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(2,2))



#model.add(Conv2D(256, (3,3)))

#model.add(Activation('relu'))

model.add(Conv2D(256, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(2,2))



#model.add(Conv2D(512, (3,3)))

#model.add(Activation('relu'))

model.add(Conv2D(512, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(1024, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(2,2))



#model.add(Conv2D(2048, (3,3)))

#model.add(Activation('relu'))

#model.add(MaxPooling2D(2,2))









model.add(Flatten())

model.add(Dense(500))

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(300))

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(100))

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(Dense(3))

model.add(Activation('softmax'))

model.summary()



target=to_categorical(y, num_classes=3)
batch_size=64

epochs=15



checkpoint = ModelCheckpoint(filepath='covid19.h5', save_best_only=True)

#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=2, mode='max')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X, target, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint])
model.save('covidLarge.h5')