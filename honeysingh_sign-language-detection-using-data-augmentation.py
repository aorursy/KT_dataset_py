# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/Sign-language-digits-dataset/"))
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D,Flatten,Dense,MaxPooling2D,Dropout,BatchNormalization
from keras.models import Sequential

X = np.load('../input/Sign-language-digits-dataset/X.npy')
Y = np.load('../input/Sign-language-digits-dataset/Y.npy')

print(X.shape)
print(Y.shape)
plt.imshow(X[5])
datagen = ImageDataGenerator(
    rotation_range = 12,
    width_shift_range = .12,
    height_shift_range = .12,
    zoom_range = .12
)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=.3)
Xtrain = Xtrain[:,:,:,np.newaxis]
Xtest = Xtest[:,:,:,np.newaxis]
datagen.fit(Xtrain)
model = Sequential()
model.add(Convolution2D(64,3,3,border_mode = 'same',input_shape = (64,64,1),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu',border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adadelta(),metrics=['accuracy'])
model.fit(Xtrain,Ytrain,epochs=3)
model.fit_generator(datagen.flow(Xtrain,Ytrain),steps_per_epoch=64,epochs=3)
