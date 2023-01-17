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
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense, BatchNormalization 
from keras.utils.np_utils import to_categorical
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
display(train.info())
display(test.info())

display(train.head())
display(test.head())
x_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
test = test.iloc[:,:].values
x_train = tf.keras.utils.normalize(x_train,axis = 1)
test = tf.keras.utils.normalize(test,axis = 1)
y_train = to_categorical(y_train, num_classes = 10)
x_train = x_train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)
model = Sequential()

model.add(BatchNormalization(input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 128, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'valid'))
model.add(Conv2D(filters = 128, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'valid'))
model.add(MaxPool2D(pool_size = 2,strides = 2))
model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'valid'))
model.add(Conv2D(filters = 256, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'valid'))
model.add(MaxPool2D(pool_size = 2,strides = 2))
model.add(BatchNormalization())

model.add(Conv2D(filters = 512, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'valid'))
model.add(Conv2D(filters = 512, kernel_size = 1, kernel_initializer = 'he_normal', activation = 'relu', padding = 'valid'))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()
a = model.fit(x_train, y_train, batch_size=100, epochs=3,verbose=1)
y = model.predict(test)
predictions = y
y = np.argmax(y, axis=1)
loss = a.history["loss"]
acc = a.history["acc"]
ep = list(range(len(loss)))
plt.plot(ep, loss)
plt.xlabel("#epochs")
plt.ylabel("loss")
plt.plot(ep, acc)
plt.xlabel("#epochs")
plt.ylabel("accuracy")
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(test)+1)]
submission['Label'] = y
submission.to_csv('submission_3_4.csv', index=False)
test = test.reshape(test.shape[0], 28, 28)
j=0
for i in range(10004, 10007):
    plt.subplot(330 + (j+1))
    j+=1
    plt.imshow(test[i], cmap = plt.cm.binary)
    plt.title(np.argmax(predictions[i]));
