' 2'# This Python 3 environment comes with many helpful analytics libraries installed
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
train_df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.head(2)
import tensorflow as tf
import matplotlib.pyplot as plt
#converting the dataframe into arrays for training data
X_train=(train_df.iloc[:,1:].values).astype('float32')
y_train=(train_df.iloc[:,0].values).astype('int32')
#converting the dataframe into arrays for testing data
X_test=(test_df.iloc[:,:].values).astype('float32')

X_train.shape,y_train.shape,X_test.shape
X_train=X_train.reshape(X_train.shape[0],28,28)
X_test=X_test.reshape(X_test.shape[0],28,28)
for i in range(6,9):
    plt.subplot(330+(i+1))
    plt.imshow(X_train[i],cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
mean_px=X_train.mean().astype(np.float32)
std_px=X_train.std().astype(np.float32)

def standardise(x):
    return (x-mean_px)/std_px
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)

#making array between 0 to 1
X_train=X_train/255
X_test=X_test/255
#designing neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
    
    
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import Callback

class mycallback(Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            print('\nreached 98% accuracy')
            self.model.stop_training=True
            
            
callback=mycallback()
    
history=model.fit(X_train,y_train,epochs=20,validation_split=0.2,verbose=1,callbacks=[callback])
history.history['accuracy']

history.history['val_accuracy']


model.summary()
predictions=model.predict(X_test)
#first three predictions of the model
for i in range(3):
    print(np.argmax(predictions[i]))

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(len(acc))

plt.plot(epochs,acc,'b',label='accuracy')
plt.plot(epochs,val_acc,'r',label='val_accuracy')
plt.legend()
plt.show()
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(epochs,loss,'b',label='loss')
plt.plot(epochs,val_loss,'r',label='val_loss')
plt.legend()
plt.show()









