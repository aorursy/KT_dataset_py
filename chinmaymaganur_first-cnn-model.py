# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
train=pd.read_csv('../input/digit-recognizer/train.csv')
train.head()
print()
test=pd.read_csv('../input/digit-recognizer/test.csv')
test.head()
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout
print(train.shape)
print(test.shape)
y_train=train['label']
#y_train.unique()
y_train=to_categorical(y_train,10)
x_train=train.drop('label',axis=1)
x_test=train.drop('label',axis=1)
x_train=x_train/255
x_test=x_test/255
#Reshape
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
#(-1,28,28,1)   nrows,height,width,channel(for  us its  gray scale)
#splitting data into train and validation
X_train,X_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=22)
print(f'x_train {X_train.shape}')
print(f'x_val {X_val.shape}')
print(f'y_train {y_train.shape}')
print(f'y_val {y_val.shape}')
model=Sequential()
#adding first conv layer
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

#adding 2nd conv layer
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 32, epochs = 1, validation_data = (X_val, y_val), verbose = 2)
history
pred=model.predict_classes(x_test)
pred
X_train.shape[0]//32
#Trying with data augumentation

train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True)
train_datagen.fit(X_train)
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=32),
                              epochs = 50, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 32)
a=pd.DataFrame(model.history.history)
import  matplotlib.pyplot as plt
plt.plot(a['loss'],label='loss')
plt.plot(a['val_loss'],color='orange',label='val_loss')
plt.legend()
plt.plot(a['accuracy'],label='accuracy')
plt.plot(a['val_accuracy'],label='val_acc')
#we can see that the cal_loss and acc are fluctuating , it may be case of overfitting, we try adding drpout/batchnormalizatiom
from tensorflow.keras.layers import BatchNormalization


model=Sequential()
#adding first conv layer
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size = (2, 2)))

#adding 2nd conv layer
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2, 2)))




model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(X_train, y_train, batch_size = 64, epochs = 4, validation_data = (X_val, y_val), verbose = 1,callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
plt.plot(model_loss['loss'])
plt.plot(model_loss['val_loss'])
predictions = model.predict_classes(x_test, verbose=0)

submission=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submission.to_csv("FirstCNN", index=False, header=True)