import numpy as np

from keras.datasets import cifar10



import PIL

import cv2



import tensorflow as tf

from skimage.transform import resize



from tensorflow.keras import models,layers,optimizers



from keras.utils import np_utils



from keras.callbacks import ModelCheckpoint



from keras.models import Sequential



from keras.layers import Dense, Conv2D, MaxPooling2D,UpSampling2D,BatchNormalization

from keras.layers import Dropout,Flatten,GlobalAveragePooling2D



from keras.applications.resnet50 import ResNet50, preprocess_input





(X_train,y_train) , (X_test,y_test) = cifar10.load_data()# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





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
#Importing the ResNet50 model



#Loading the ResNet50 model with pre-trained ImageNet weights

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

conv_base.summary()
X_train = X_train/255.0

X_test = X_test/255.0



y_train = np_utils.to_categorical(y_train,10)

y_test = np_utils.to_categorical(y_test,10)



model = Sequential()

model.add(UpSampling2D((2,2)))

model.add(UpSampling2D((2,2)))

model.add(UpSampling2D((2,2)))

model.add(conv_base)

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])



history = model.fit(X_train, y_train, epochs=5, batch_size=20, validation_data=(X_test, y_test))
model.evaluate(X_test, y_test)

history