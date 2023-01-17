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
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import  shuffle
import os
import tensorflow as tf
from random import shuffle
IMG_SIZE=224
X=[]
Y=[]
training_data=[]
test_data=[]
X_t=[]
Y_t=[]
path="/kaggle/input/dogs-cats-images/dataset/training_set/"
for i in os.listdir(path+'cats/'):
    img=cv2.imread(path+'cats/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    training_data.append(tuple([img,1]))
    #X.append(img)
    #Y.append(1)
for i in os.listdir(path+'dogs/'):
    img=cv2.imread(path+'dogs/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    training_data.append(tuple([img,0]))
    #X.append(img)
    #Y.append(0)

shuffle(training_data)
for features,labels in training_data:
    X.append(features)
    Y.append(labels)

del training_data

path="/kaggle/input/dogs-cats-images/dataset/test_set/"
for i in os.listdir(path+'cats/'):
    img=cv2.imread(path+'cats/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    test_data.append(tuple([img,1]))
    #X.append(img)
    #Y.append(1)
for i in os.listdir(path+'dogs/'):
    img=cv2.imread(path+'dogs/'+i)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    test_data.append(tuple([img,0]))
    #X.append(img)
    #Y.append(0)

shuffle(test_data)
for features,labels in test_data:
    X_t.append(features)
    Y_t.append(labels)

del test_data

X=np.array(X)
#X=X/225.0
Y=np.array(Y)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
X.shape

X_t=np.array(X_t)
#X_t=X-t/225.0
Y_t=np.array(Y_t)
X_t=np.array(X_t).reshape(-1,IMG_SIZE,IMG_SIZE,3)
X_t.shape
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.95):
            self.model.stop_training = True

callback=myCallBack()
        
model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape=X.shape[1:],activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



model.fit(X,Y,epochs=30,validation_data=(X_t,Y_t),callbacks=[callback])
model.save('saved_model.h5')
