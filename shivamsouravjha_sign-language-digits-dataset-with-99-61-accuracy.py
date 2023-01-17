# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from tensorflow import keras
import cv2
from keras.models import load_model
import math
import numpy as np

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,Adam,SGD,Adagrad,Adadelta,Adamax,Nadam
from keras.applications import xception
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x_l = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')    #loading X 
Y_l = np.load('/kaggle/input/sign-language-digits-dataset/Y.npy')   #loading y
img_size = 64
plt.subplot(1,2,1)
plt.imshow(x_l[700].reshape(img_size,img_size))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x_l[1900].reshape(img_size,img_size))
plt.axis('off')

X=x_l
Y=Y_l
X = X.reshape(-1,64,64,1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
test_size = 0.25
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
print("x_train shape",X_train.shape)
print("x_test shape",X_test.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_test.shape)
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation= 'relu',input_shape=(64,64,1)))          #adding convolution
model.add(BatchNormalization())                     #normalising the batch
model.add(MaxPool2D(pool_size=(2,2)))                            #max pooling 
model.add(Dropout(0.2))                                    #dropping ou t20%
model.add(Conv2D(64,(3,3),padding='same',activation= 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(64,(3,3),padding='same',activation= 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(64,(3,3),padding='same',activation= 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())                  #flatten the outpt by above layers
model.add(Dense(256,activation= 'relu'))                 #dense layer 1
model.add(Dense(128,activation= 'relu'))
model.add(Dense(64,activation= 'relu'))

model.add(Dense(10,activation= 'softmax'))                 #outputting from dense layer on 10 selections

model.summary()
from keras.optimizers import RMSprop,Adam,SGD,Adagrad,Adadelta,Adamax,Nadam

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99)
# Compile the model
model.compile(optimizer = optimizer , loss = 'categorical_crossentropy', metrics=["accuracy"])
# fitting

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image("model.png")
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.994):
            print("\nReached 99.0% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

history = model.fit(X_train,Y_train,epochs=225,validation_data=(X_test,Y_test),callbacks = [callbacks])
scores = model.evaluate(X_test, Y_test, verbose=0)
print("{}: {:.2f}%".format("accuracy", scores[1]*100))
model.save('Sign Language Digits Dataset')
fname = "../input/testingstuff/images (1).jpg"
img = cv2.imread(fname,0)

img = cv2.resize(img,(64,64))


plt.figure(figsize=(10,6))
plt.imshow(img)

img = np.reshape(img,[1,64,64,1])

model = load_model('Sign Language Digits Dataset')

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])


classes = model.predict_classes(img)

print (classes)
