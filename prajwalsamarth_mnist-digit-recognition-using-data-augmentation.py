import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.head()
target=train['label']

train=train.drop('label',axis=1)

train=train.values.reshape(-1,28,28)

train=train/255

plt.imshow(train[1,:,:])

print(target[1])
from sklearn.model_selection import train_test_split

x_train ,  x_test ,y_train, y_test= train_test_split(train,target,test_size=0.2,random_state=7)

print(x_train.shape)

print(x_test.shape)
test=test.values.reshape(-1,28,28)

test=test/255

test.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,BatchNormalization,Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator
def get_model(input_shape,drop_rate):

  model=Sequential([

                    Conv2D(32,(3,3),input_shape=input_shape,padding='SAME',activation='relu'),

                    Conv2D(32,(3,3),padding='SAME',activation='relu'),

                    MaxPool2D((2,2)),

                    Dropout(drop_rate),

                    BatchNormalization(),

                    Conv2D(64,(3,3),padding='SAME',activation='relu'),

                    Conv2D(64,(3,3),padding='SAME',activation='relu'),

                    MaxPool2D((2,2)),

                    Dropout(drop_rate),

                    BatchNormalization(),

                    Flatten(),

                    Dense(128,activation='relu'),

                    Dense(10,activation='softmax')

  ])

  return model

mod=get_model([28,28,1],0.3)

mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
data_aug=ImageDataGenerator(rotation_range=20,shear_range=0.4,zoom_range=[0.75,1.3])

history=mod.fit_generator(data_aug.flow(x_train[...,np.newaxis],y_train,batch_size=100),epochs=500,

                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15),

                                     tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,patience=15)],validation_data=(x_test[...,np.newaxis],y_test))
pred=mod.predict(test[...,np.newaxis])

labels=np.argmax(pred,axis=1)

results=pd.DataFrame({'label':labels},index=range(1,test.shape[0]+1))
#results.to_csv('result3.csv')
hist=pd.DataFrame(history.history)

hist.plot(y='accuracy')

plt.plot(hist['val_accuracy'])
hist.plot(y='loss')

plt.plot(hist['val_loss'])