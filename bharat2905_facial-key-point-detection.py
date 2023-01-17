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
print(os.listdir("../input/training"))
training_data=pd.read_csv('../input/training/training.csv')

training_data.head()
training_data.isnull().sum()
training_data=training_data.dropna(axis=0)
training_data.isnull().sum()
training_data.iloc[2000]
rows=training_data.axes[0].tolist()
image = []

for row  in rows:

    img = training_data['Image'][row].split(' ')

    img = ['0' if x == '' else x for x in img]

    image.append(img)
len(image)
image_list = np.array(image,dtype = 'float')

X_train = image_list.reshape(-1,96,96,1)
len(image_list)
len(X_train)
training = training_data.drop('Image',axis = 1)



y_train = []

for i in range(len(rows)):

    y = training.iloc[i,:]



    y_train.append(y)

y_train = np.array(y_train,dtype = 'float')
y_train = y_train/96
import matplotlib.pyplot as plt

plt.imshow(X_train[2].reshape(96,96),cmap='gray')

for i in range(15):

    plt.plot(96*y_train[2][2*i],96*y_train[0][2*i+1],'ro')

plt.show()
from keras.layers import Conv2D,Dropout,Dense,Flatten

from keras.models import Sequential

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same',input_shape=(96,96,1),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(32, (3,3), padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))





model.add(Convolution2D(96, (3,3), padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same',activation='relu'))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same',activation='relu'))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same',activation='relu'))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30,activation='relu'))

model.summary()
model.compile(optimizer='adam', 

              loss='mean_squared_error',

              metrics=['mae'])
model.fit(X_train,y_train,epochs = 500,batch_size = 256,validation_split = 0.2)
im=X_train[500]
im=im.reshape(1,96,96,1)
pred=np.array(model.predict(im))
pred[0]
plt.imshow(im.reshape(96,96),cmap='gray')

for i in range(15):

    plt.plot(96*pred[0][2*i],96*pred[0][2*i+1],'ro')

plt.show()