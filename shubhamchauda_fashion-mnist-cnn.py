# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)r
from tensorflow import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_data =pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
train_data.head()
y__train = train_data['label']
y__test = test_data['label']
X_train = train_data.drop('label',axis=1)
X_train = (X_train.to_numpy()).reshape((len(train_data),28,28,1))/255
X_test = test_data.drop('label',axis = 1)
X_test = (X_test.to_numpy()).reshape((len(test_data),28,28,1))/255
X_train, X_val, y__train, y_val = train_test_split(X_train, y__train, test_size=0.35, random_state=42)
y_train =keras.utils.to_categorical(y__train, num_classes=10)
y_val =keras.utils.to_categorical(y_val, num_classes=10)
y_test =keras.utils.to_categorical(y__test, num_classes=10)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
model = keras.Sequential()
model.add(Conv2D(filters = 64,kernel_size = (3,3), activation = 'relu' ,input_shape = (28,28,1),padding = 'same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 64,kernel_size = (3,3), activation = 'relu',padding = 'same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters = 32,kernel_size = (3,3), activation = 'relu',padding = 'same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units =512,activation = 'relu'))
model.add(Dense(units =128,activation = 'relu'))
model.add(Dense(units = 10,activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'] )
history = model.fit(X_train,y_train, verbose = 1, epochs = 10 , validation_data = (X_val,y_val))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
model.evaluate(X_test,y_test)

plt.plot(history.history['loss'], label='accuracy')
plt.plot(history.history['val_loss'], label = 'val_accuracy')