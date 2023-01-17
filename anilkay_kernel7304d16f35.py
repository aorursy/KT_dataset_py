# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i=0

for dirname, _, filenames in os.walk('/kaggle/input'):

    break

    if i>3:

        break

    i=i+1    

    for filename in filenames:

        break

        #print(os.path.join(dirname, filename))

print("bir")

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/celeba-dataset/list_attr_celeba.csv")
data.head()
attractive=data["Attractive"]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

attr2=le.fit_transform(attractive)
imagepath="/kaggle/input/celeba-dataset/img_align_celeba/"

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)



image_generator = datagen.flow_from_directory(

        imagepath,

        target_size=(150, 150),

        batch_size=32

       )
image_generator.classes=attr2
import skimage.io as imio

deneme=imio.imread("/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/000001.jpg")

imio.imshow(deneme)
set(attr2)
image_generator.class_mode="binary"
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.constraints import maxnorm

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils



model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=(150,150,3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dropout(0.2))



model.add(Dense(256, kernel_constraint=maxnorm(3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(1))

model.add(Activation('sigmoid'))



optimizer = 'Adam'



model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
ResultsTrain = model.fit_generator(image_generator, epochs=2,verbose=1)
image_generator.filenames