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
import os

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout

import pandas as pd

import numpy as np
dir="../input"

os.chdir(dir)
train_x=pd.read_csv("fashion_train.csv",header=0)

train_y=pd.read_csv("fashion_train_labels.csv",header=0)

test_x=pd.read_csv("fashion_test.csv",header=0)

test_y=pd.read_csv("fashion_test_labels.csv",header=0)
labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

train_x=train_x/255

train_y=train_y/255

test_x=test_x/255

test_y=test_y/255
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(train_x.head())
print(test_x.head())
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(np.array(train_x.iloc[0]).reshape((28,28)),cmap='gray')
plt.imshow(np.array(train_x.iloc[1]).reshape((28,28)),cmap='gray')
plt.imshow(np.array(train_x.iloc[2]).reshape((28,28)),cmap='gray')
plt.imshow(np.array(train_x.iloc[9]).reshape((28,28)),cmap='gray')
from keras.wrappers.scikit_learn import KerasClassifier

from keras.regularizers import l1,l2
model=tf.keras.models.Sequential([

    tf.keras.layers.Dense(units=30,input_dim=784,kernel_regularizer=l2(0.001)),

    tf.keras.layers.Dense(units=1000,activation=tf.nn.relu),

    tf.keras.layers.Dense(units=1000,activation=tf.nn.relu),

    tf.keras.layers.Dense(units=500,activation=tf.nn.relu),

    tf.keras.layers.Dense(units=200,activation=tf.nn.relu),

    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)

])
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
m=model.fit(train_x,train_y,epochs=10,validation_split=0.20)
test_loss,test_acc=model.evaluate(test_x,test_y)

print("Test Accuracy:",(test_acc*100))
p=model.predict(np.array(test_x.loc[0]).reshape(1,784))

p
np.argmax(p)
plt.imshow(np.array(test_x.loc[0]).reshape((28,28)),cmap='gray')
predict=model.predict(test_x)

predict[1]
np.argmax(predict[1])
labels[0]
plt.plot(m.history['acc'],color='green')

plt.plot(m.history['val_acc'],color='blue')

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train','test'],loc='upper_left')

plt.show()
plt.plot(m.history['loss'],color='green')

plt.plot(m.history['val_loss'],color='blue')

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train','test'],loc='upper_left')

plt.show()
models=Sequential()

models.add(Dense(units=30,input_dim=784,kernel_regularizer=l2(0.001)))

models.add(Activation('relu'))

models.add(Dense(units=1000))

models.add(Activation('relu'))

models.add(Dense(units=1000))

models.add(Activation('relu'))

models.add(Dense(units=500))

models.add(Activation('relu'))

models.add(Dense(units=200))

models.add(Activation('relu'))

models.add(Dense(units=10))

models.add(Activation('softmax'))
models.summary()

models.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
x=np.array(train_x)

import keras

y=keras.utils.to_categorical(np.array(train_y),10)

m2=models.fit(x,y,epochs=10,validation_split=0.20)
p2=models.predict(np.array(test_x.loc[0]).reshape(1,784))

p2
np.argmax(p2)
plt.imshow(np.array(test_x.loc[5]).reshape((28,28)),cmap='gray')
labels[0]
p3=models.predict_proba(np.array(test_x.loc[8]).reshape(1,784))
np.argmax(p3)
labels[0]
plt.plot(m2.history['acc'])

plt.plot(m2.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train','test'],loc='upper_left')

plt.show()
plt.plot(m2.history['loss'])

plt.plot(m2.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train','test'],loc='upper_left')

plt.show()