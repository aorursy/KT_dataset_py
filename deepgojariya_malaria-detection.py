# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
train = pd.read_csv('/kaggle/input/malaria-detection-train/train.csv')
test = pd.read_csv('/kaggle/input/malaria-detection-test/test.csv')
train.head()
import matplotlib.pyplot as plt
X_train = train.drop(['label'],axis=1).values
y_train = train['label'].values
X_test = test.drop(['label'],axis=1).values
y_test = test['label'].values
plt.imshow(X_train[2000].reshape(50,50),cmap='gray')
print(y_train[2000])
plt.imshow(X_train[20000].reshape(50,50),cmap='gray')
print(y_train[20000])
X_train = X_train.reshape(X_train.shape[0],50,50,1).astype('float32')
X_train = X_train / 255.0

X_test = X_test.reshape(X_test.shape[0],50,50,1).astype('float32')
X_test = X_test / 255.0
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
y_train[0]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=3,padding='same',activation='relu',input_shape=(50,50,1)))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=200,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=2,activation='softmax'))
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train_1,batch_size=50,epochs=20)
predictions = model.evaluate(X_test,y_test)
index=100
plt.imshow(X_test[index].reshape(50,50),cmap='gray')
print(y_test[index])
X_test[index]
X_test[0].shape
import numpy as np
test_image = np.expand_dims(X_test[4200], axis = 0)
result = model.predict(test_image)
result = result[0].argmax()

print(result)
if result==0:
    pred='Infected'
else:
    pred='uninfected'
pred
