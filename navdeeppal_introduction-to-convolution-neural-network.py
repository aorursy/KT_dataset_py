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
from keras import layers 
from keras import models
convnet_model = models.Sequential()
convnet_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
convnet_model.add(layers.MaxPooling2D((2, 2)))
convnet_model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
convnet_model.add(layers.MaxPooling2D((2, 2)))
convnet_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

convnet_model.summary()
convnet_model.add(layers.Flatten())
convnet_model.add(layers.Dense(64, activation='relu'))
convnet_model.add(layers.Dense(10, activation='softmax'))

convnet_model.summary()
train_path = "/kaggle/input/digit-recognizer/train.csv"
test_path = "/kaggle/input/digit-recognizer/test.csv"

# reading data

import pandas as pd
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print("Shape of trainig set :", train.shape)
print("Shape of test set :", test.shape)
train_set = train[:30000]
validation_set = train[30000:]
# reshaping the datasets to 3D tensors
train_labels = train_set.iloc[:,0]
train_images = train_set.iloc[:,1:]
train_images = train_images.values.reshape((30000, 28, 28, 1))


test_labels = validation_set.iloc[:,0]
test_images = validation_set.iloc[:,1:]
test_images = test_images.values.reshape((12000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
convnet_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
convnet_model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = convnet_model.evaluate(test_images, test_labels)
test_acc
train_labels = train.iloc[:,0]
train_images = train.iloc[:,1:]
train_images = train_images.values.reshape((42000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
convnet_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
convnet_model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_images = test.values.reshape((28000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
predictions = convnet_model.predict(test_images)
predictions = np.asarray([np.argmax(prediction) for prediction in predictions])
predictions.shape
df_predictions = pd.DataFrame(predictions).rename(columns={0: "Label"})
df_predictions.index.names = ['ImageId']
df_predictions.index += 1
df_predictions.head()
df_predictions.shape
df_predictions.to_csv("predictions.csv")
from IPython.display import FileLink
FileLink(r'predictions.csv')