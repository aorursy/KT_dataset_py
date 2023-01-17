#Convolutional Neural Network on MNIST handwritten digit dataset
#importing libraries

import keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np 
data = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
data.head()
#reshaping into 28X28 array

data.iloc[3,1:].values.reshape(28,28).astype('uint8')
#preprocessing data 
#Storing Pixel array in form length width and channel in df_x

df_x = data.iloc[:,1:].values.reshape(len(data),28,28,1)



#Storing the labels in y

y = data.iloc[:,0].values
#Converting labels to categorical features



df_y = keras.utils.to_categorical(y,num_classes=10)
df_x = np.array(df_x)

df_y = np.array(df_y)
#lables

y
#categorical labels

df_y
df_x.shape
#test train split



x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#CNN model

model = Sequential()

model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
model.summary()
#fitting it with just 100 images for testing 



model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs = 15 ,batch_size=64)
model.evaluate(x_test,y_test)
#imporove accuracy by more epocs till the loss is almost same