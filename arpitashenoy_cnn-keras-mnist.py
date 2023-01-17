from __future__ import print_function,division

from builtins import range



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Conv2D, Dense, Activation, BatchNormalization, MaxPool2D, Flatten, Dropout

from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data = data.as_matrix()

np.random.shuffle(data)
x = data[:, 1:].reshape(-1,28,28,1)/255.0

y = data[:, 0].astype(np.int32)
#get shape

N  = len(y)

k = len(set(y))
y = to_categorical(y)
model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), filters = 32, kernel_size=(3,3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D())
model.add(Conv2D(filters = 64, kernel_size=(3,3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D())
model.add(Flatten())

model.add(Dense(units=300))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(units=k))

model.add(Activation('softmax'))
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
r = model.fit(x,y, validation_split=0.33, epochs=8, batch_size=32)
x_test = pd.read_csv('../input/test.csv')
x_test = x_test.as_matrix()
x_test = x_test[:,:].reshape(-1,28,28,1)/255.0
result = model.predict(x_test)



result = np.argmax(result, axis = 1)



result = pd.Series(result, name = 'Label')



plt.plot(r.history['loss'], label = 'loss')

plt.plot(r.history['val_loss'], label = 'val_loss')

plt.legend()

plt.show()
plt.plot(r.history['acc'], label = 'accuracy')

plt.plot(r.history['val_acc'], label = 'Validation_accuracy')

plt.legend()

plt.show()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)



export_csv = submission.to_csv("cnn_keras_mnist.csv",index=False)