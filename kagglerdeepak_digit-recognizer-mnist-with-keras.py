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
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
# All pixel values - all rows and column 1 (pixel0) to column 785 (pixel 783)
x_train = (train.iloc[:,1:].values).astype('float32') 
# Take a look at x_train
x_train
# Labels - all rows and column 0
y_train = (train.iloc[:,0].values).astype('int32') 

# Take a look at y_train
y_train
test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
x_test = test.values.astype('float32')
x_test
from keras.utils import to_categorical
num_classes = 10

# Normalize the input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape input data from (samples, 28, 28) to (samples, 28*28)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes)

# Take a look at the dataset shape after conversion with keras.utils.to_categorical
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Dropout(rate=0.5))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128)
predictions = model.predict_classes(x_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("mnist_tfkeras.csv", index=False, header=True)
print(os.listdir(".."))

