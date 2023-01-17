# This Python 3 environment comes with many helpful analytics libraries installed# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow.keras

from tensorflow.keras.layers import Dense,Dropout

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir())



# Any results you write to the current directory are saved as output.
df_test=pd.read_csv("../input/test.csv")

df_train=pd.read_csv("../input/train.csv")
def one_hot_conv(a, a_len):

    a_oh = np.zeros((a.shape[0],a_len))

    for i,j in zip(range(a.shape[0]),a):

        a_oh[i,j]=1

    return a_oh
# Get training data uploaded into Numpy array

X = df_train.values

Y = X[:,0]

X = X[:,1:]

# get test data uploaded into Numpy array

X_pred = df_test.values

Y_pred = np.zeros((X.shape[0]))

# normalize X_train and X_test

X, X_pred = X / 255 , X_pred / 255

#Split data for test and train

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
print(os.listdir())
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
# reshape the images to 28X28 to use Conv net

X_train1=X_train.reshape(29400,28,28,1)

X_test1=X_test.reshape(12600,28,28,1)

Y_train1=one_hot_conv(Y_train,10)

Y_test1=one_hot_conv(Y_test,10)
#Data Augumentation

datagen=tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,

    featurewise_std_normalization=False,

    rotation_range=10,

    zoom_range=0.1,

    height_shift_range=0.1,

    width_shift_range=0.1)



datagen.fit(X_train1)

datagen.fit(X_test1)



train_gen = datagen.flow(X_train1, Y_train1, batch_size=32)

test_gen = datagen.flow(X_test1, Y_test1, batch_size=32)
# create a Conv 2D model

model_conv=tf.keras.models.Sequential([

    (tf.keras.layers.Conv2D(input_shape=(28,28,1),filters=32,kernel_size=(3,3),strides=1,activation="relu", padding='same')),

    (tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu")),

    (tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)),

    (tf.keras.layers.BatchNormalization(axis=3)),

    (tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation="relu")),

    (tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation="relu")),

    (tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)),

    (tf.keras.layers.BatchNormalization(axis=3)),

    (tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation="relu")),

    (tf.keras.layers.BatchNormalization(axis=3)),

    (tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)),

    (tf.keras.layers.Flatten()),

    (tf.keras.layers.Dense(256,activation="relu")),

#     (tf.keras.layers.Dense(128,activation="relu")),

#     (tf.keras.layers.Dense(64,activation="relu")),

    (tf.keras.layers.Dense(10,activation="softmax"))

])



model_conv.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=['accuracy'])

model_conv.summary()
fit1=model_conv.fit_generator(train_gen,validation_data = test_gen,epochs=30)

# model_conv.evaluate(test_gen)
import matplotlib.pyplot as plt

plt.plot(fit1.history['acc'])

plt.plot(fit1.history['val_acc'])

plt.show()



plt.plot(fit1.history['loss'])

plt.plot(fit1.history['val_loss'])

plt.show()
# evaluate the model against the original version(before data aug) for test data set



X_test2=X_test.reshape(12600,28,28,1)

Y_test2=one_hot_conv(Y_test,10)



model_conv.evaluate(X_test2,Y_test2)
X_pred1=X_pred.reshape(-1,28,28,1)

Y_pred1=model_conv.predict(X_pred1)
Y_pred_final=np.zeros((28000,2))



for i in range(28000):

    Y_pred_final[i,0]=i

    Y_pred_final[i,1]=np.argmax(Y_pred1[i])



Y_pred_final[0:10]
df_result=pd.DataFrame(Y_pred_final,columns=['ImageID','Label'])

df_result.head()
df_result.to_csv('submission.csv',index=False)