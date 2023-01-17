import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import layers
from keras import models
from keras.utils.np_utils import to_categorical
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#loading dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
test.shape

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()
#labels
y_train = train['label']

x_train = train.drop(labels = ["label"],axis = 1)
x_train = x_train/255.0
test = test/255.0

x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
from keras.utils import to_categorical

y_train = to_categorical(y_train)
model.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['acc'])
model.fit(x_train, y_train, epochs =5, batch_size = 128)