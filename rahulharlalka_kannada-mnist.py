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

from tensorflow.keras import layers

import numpy as np

import pandas as pd

import os

import cv2

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts

import seaborn as sns

x_train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

y_train=x_train[['label']]

y_train.shape

x_train.drop(labels='label',axis=1,inplace=True)
x_train=np.array(x_train).reshape(len(x_train),28,28,1)

x_train=x_train/255.0

print("shape of training data=",x_train.shape)

y_train=np.array(y_train)

print("shape of training labels=",y_train.shape)

x_train,x_test,y_train,y_val=tts(x_train,y_train,test_size=0.2,random_state=42)
print("training data shape=",x_train.shape)

print("testing data shape=",x_test.shape)

print("training labels shape=",y_train.shape)

print("testing labels shape=",y_val.shape)

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(4096)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_val)).shuffle(10000).batch(4096)
model=keras.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer='l2'))



model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer='l2'))

model.add(layers.BatchNormalization())





model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer='l2'))

model.add(layers.MaxPooling2D((2,2)))





model.add(layers.Flatten())



model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.5))





model.add(layers.Dense(10,activation='softmax'))

model.summary()
from tensorflow import keras

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.001)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

tf.keras.backend.clear_session()

model.fit(train_data,epochs=10,callbacks=[reduce_lr])
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.fit(train_data,epochs=20,initial_epoch=10,callbacks=[reduce_lr])
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.00001),metrics=['accuracy'])
model.fit(train_data,epochs=40,initial_epoch=20,callbacks=[reduce_lr])
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.00001,amsgrad=True),metrics=['accuracy'])

model.fit(train_data,initial_epoch=40,epochs=60,callbacks=[reduce_lr])
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.000005,amsgrad=True),metrics=['accuracy'])

model.fit(train_data,initial_epoch=60,epochs=80)
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0000001,amsgrad=True),metrics=['accuracy'])

model.fit(train_data,initial_epoch=80,epochs=100)
model.evaluate(test_data)


plt.plot(model.history.history['accuracy'])

#plt.plot(model.history.history['val_accuracy'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend(['training_acc'])

plt.show()

plt.savefig('accuracy_plots.jpg')
model.history.history.keys()
plt.plot(model.history.history['loss'])

#plt.plot(model.history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.title('loss of data over epochs')

plt.legend(['training_loss','testing_loss'])

plt.show()

plt.savefig('loss_plts.jpg')
submit_x=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

ids=submit_x[['id']]

submit_x.drop(labels='id',axis=1,inplace=True)

submit_x=np.array(submit_x).reshape(len(submit_x),28,28,1)

submit_x=submit_x/255.0

submit_y=model.predict(submit_x)

Y=[]

for row in submit_y:

    Y.append(np.argmax(row))

Y=np.array(Y)

Y=pd.DataFrame(Y)

Y.rename(columns={0:'label'},inplace=True)
final_file=pd.concat([ids,Y],axis=1)
final_file.to_csv('submission.csv',index=False)