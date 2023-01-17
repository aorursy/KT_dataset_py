# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train=pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
test=pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Ytrain=train["label"]
train=train.drop(["label"],axis=1)

Ytest=test["label"]
Xtest=test.drop(["label"],axis=1)

# print(Ytrain.value_counts())
print(Ytrain.sort_values(ascending=True).unique())
print(Ytest.sort_values(ascending=True).unique())
# Xtrain.isnull().any().describe()
# Xtest.isnull().any().describe()

train=train/255
Xtest=Xtest/255

Ytrain=to_categorical(Ytrain,num_classes=25)
Ytest=to_categorical(Ytest,num_classes=25)
train=train.values.reshape(-1,28,28,1)
Xtest=Xtest.values.reshape(-1,28,28,1)

print(train.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)
random_seed = 1
Xtrain, Xval, Ytrain, Yval = train_test_split(train, Ytrain, test_size = 0.1, random_state=random_seed)
print(Xtrain.shape)
print(Ytrain.shape)
print(Xval.shape)
print(Yval.shape)
model = keras.Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(25, activation = "softmax"))

model.summary()
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
batch_size = 64
epochs=10
history = model.fit(Xtrain, Ytrain, batch_size = batch_size, epochs = epochs, 
          validation_data = (Xval, Yval), verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.evaluate(Xtest,Ytest)
pred=model.predict(Xtest)
from sklearn.metrics import accuracy_score
accuracy_score(Ytest, pred.round())
