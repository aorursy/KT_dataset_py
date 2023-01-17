# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from keras.utils import to_categorical 

from keras import backend as K

from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras.layers import Dropout

from keras.layers.core import Activation



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/sign_mnist_train.csv')

print(df.shape)

df.head()

train=df.values[0:,1:]

labels = df.values[0:,0]

labels = to_categorical(labels)

sample = train[1]

plt.imshow(sample.reshape((28,28)))
print(train.shape,labels.shape)

#normalizing the dataset

train=train/255

train=train.reshape((27455,28,28,1))

plt.imshow(train[1].reshape((28,28)))

print(train.shape,labels.shape)
model = Sequential()

model.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = (28,28,1),activation = 'relu',padding = 'same'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,kernel_size = (3,3),padding = 'same',activation = 'relu'))

model.add(Flatten())

model.add(Dense(64,activation = 'relu'))

model.add(Dense(25,activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

h=model.fit(train, labels, validation_split=0.3, epochs=6,batch_size=64)
plt.plot(h.history['acc'])

plt.plot(h.history['val_acc'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title('Model Loss')

plt.show()

LOC = 25

sample = train[LOC]

plt.imshow(sample.reshape((28,28)))

lbl=labels[LOC]

print(list(lbl).index(1))
sample=sample.reshape((1,28,28,1))

res=model.predict(sample)

res=list(res[0])

mx=max(res)

print(res.index(mx))