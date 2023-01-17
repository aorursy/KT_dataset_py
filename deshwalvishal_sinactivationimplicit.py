import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers 

import numpy as np

import keras.optimizers

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from keras.layers import Dense
from keras.layers import Activation

from keras import backend as K

from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):

  return (tf.math.sin(x))
get_custom_objects().update({'custom_activation':Activation(custom_activation)})
img=plt.imread('../input/sinsquareimage/chess.jpg')

plt.imshow(img)

print(img.shape)
model = keras.Sequential()



model.add(Dense(128, input_dim=3))

model.add(Activation(custom_activation, name='SinActivation'))

model.add(Dense(64))

model.add(Activation(custom_activation, name='SinActivation1'))

model.add(Dense(256, activation="softmax", name="softmax"))

input_arr=np.reshape(img, (1,img.shape[0]*img.shape[1]*img.shape[2]))

X=np.zeros((input_arr.shape[1],3))

y=np.zeros((input_arr.shape[1]))

#print(input_arr.shape)

index=0

for i in range(img.shape[0]):

    for j in range(img.shape[1]):

        for k in range(img.shape[2]):

            X[index,0]=i

            X[index,1]=j

            X[index,2]=k

            y[index]=img[i,j,k]

            index+=1


print(X.shape,y.shape)

#y =tf.keras.utils.to_categorical(y,256) 

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#y =tf.keras.utils.to_categorical(y,256) 

#y.shape

y_train =tf.keras.utils.to_categorical(y_train,256) 

y_test =tf.keras.utils.to_categorical(y_test,256) 

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0020),metrics=['accuracy']) 

epochs = 50

history = model.fit(X_train, y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=128,verbose=2) 
y_out = model.predict_classes(X_test)

np.max(y_out)

y_out[X_test.shape[0]-1]

y_train1=np.argmax(y_train, axis=1)

print(y_train1)
img1=np.zeros((img.shape[0], img.shape[1], img.shape[2]))

print(img1.shape)

for index in range(X_test.shape[0]):

    img1[int(X_test[index][0]),int(X_test[index][1]),int( X_test[index][2])]=y_out[index]

for index in range(X_train.shape[0]):

    img1[int(X_train[index][0]),int(X_train[index][1]),int( X_train[index][2])]=y_train1[index]

print(img1.shape)

plt.imshow(img1)