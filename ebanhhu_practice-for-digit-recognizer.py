# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Load the data
train_images = pd.read_csv("../input/train.csv")
test_images = pd.read_csv("../input/test.csv")

print(train_images.shape)
print(test_images.shape)

# Process the data
train_data = (train_images.iloc[:,1:].values).astype('float32')
train_targets = train_images.iloc[:,0].values.astype('int32')
print(train_data.shape)
print(train_targets.shape)

test_data = test_images.values.astype('float32')
print(test_data.shape)

train_data /= 255
test_data /= 255

train_targets = to_categorical(train_targets)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_targets, epochs=10, batch_size=128)

predictions = model.predict(test_data)
predictions = np.argmax(predictions, axis = 1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)

