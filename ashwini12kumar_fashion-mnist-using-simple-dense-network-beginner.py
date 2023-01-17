# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train_data=pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test_data=pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#### DATA PRE-PROCESSING
train_data.head()
train_data.shape
import tensorflow as tf
from tensorflow import keras
# from keras import layers
train_data=train_data.to_numpy()
train_images=train_data[::,1::]
train_labels=train_data[::,0]
train_labels=keras.utils.to_categorical(train_labels, num_classes=None, dtype="float32")
train_labels[0]
train_images=train_images/255
test_data.shape
test_data=test_data.to_numpy()
test_images=test_data[::,1::]
test_labels=test_data[::,0]
test_labels=keras.utils.to_categorical(test_labels, num_classes=None, dtype="float32")
test_labels.shape
#model
# A simple dense network

model=keras.Sequential()
model.add(keras.layers.Input(shape=(784,)))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, validation_split=0.2, shuffle=True, epochs=8)
prediction=model.predict(test_images, batch_size=100)
print(prediction.shape)
print(prediction[0])
model.evaluate(x=test_images, y=test_labels)
# Now let's plot the confusion matrix. For this, lets change our predicted and actual labels from one-hot form

y_test=np.argmax(test_labels, axis=1)
y_predict=np.argmax(prediction, axis=1)
print(test_labels[0])
print(y_test.shape)
print(y_predict.shape)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_predict, y_test)
model.save('fashion_mnist_model')