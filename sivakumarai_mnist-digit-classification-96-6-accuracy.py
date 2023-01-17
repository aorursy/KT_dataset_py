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
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import numpy as np
train_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv',na_values='?')
test_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_dataset.head()
train_dataset.shape

test_dataset.head()
test_dataset.shape
train_dataset.info()
train_x = train_dataset.sample(frac=0.8,random_state=10)
eval_x = train_dataset.drop(train_x.index)
train_y = train_x.pop('label')
eval_y = eval_x.pop('label')
print(train_x.shape)
print(train_y.shape)
print(eval_x.shape)
print(eval_y.shape)

train_x.head()
eval_x.head()

train_x = train_x / 255.0
eval_x = eval_x / 255.0
test_x = test_dataset / 255.0
train_x.head()
model = keras.Sequential()
#model.add(keras.layers.Conv2D(32,(3,3),strides=1, padding='valid',activation='relu'))
#model.add(keras.layers.MaxPooling2D((2,2)))
#model.add(keras.layers.Conv2D(64,(3,3),strides=1, padding='valid',activation='relu'))
#model.add(keras.layers.MaxPooling2D((2,2)))
#model.add(keras.layers.Conv2D(64,(3,3),strides=1, padding='valid',activation='relu'))
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_x = np.array(train_x)
train_x = train_x.reshape(33600,28,28)
train_x.shape

eval_x = np.array(eval_x)
eval_x = eval_x.reshape(8400,28,28)
eval_x.shape
test_x = np.array(test_x)
test_x = test_x.reshape(28000,28,28)
test_x.shape
model.fit(train_x,train_y,epochs=10)

eval_loss, eval_acc = model.evaluate(eval_x,eval_y)
print("loss:",eval_loss)
print("accuracy:", eval_acc)
predictions = model.predict(test_x)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_x[i])
    plt.xlabel(np.argmax(predictions[i]))
plt.show()