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
import cv2

%matplotlib inline

import matplotlib.pyplot as plt

import keras

import keras.backend as K

from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization,Activation

from keras.layers.merge import add

from keras.models import Model,Sequential

from sklearn.model_selection import train_test_split
import glob
uninfected = glob.glob("../input/cell_images/cell_images/Uninfected/*.png")

parasitized = glob.glob("../input/cell_images/cell_images/Parasitized/*.png")
os.listdir("../input/cell_images/cell_images/")
uninfected_imgs = [cv2.imread(i,0) for i in uninfected]

parasitized_imgs = [cv2.imread(i,0) for i in parasitized]
uninfected_imgs = [cv2.resize(i,(100,100),interpolation = cv2.INTER_AREA).reshape((100,100,1)) for i in uninfected_imgs]

parasitized_imgs = [cv2.resize(i,(100,100),interpolation = cv2.INTER_AREA).reshape((100,100,1)) for i in parasitized_imgs]
uninfected_imgs = np.array(uninfected_imgs)

parasitized_imgs = np.array(parasitized_imgs)
print(uninfected_imgs.shape)

print(parasitized_imgs.shape)
a = [[i,np.array([0,1])] for i in uninfected_imgs]

a += [[i,np.array([1,0])] for i in parasitized_imgs]

print(len(a))

a = np.array(a)

print(a.shape)
for i in range(25):

    np.random.shuffle(a)
X = np.array([i[0] for i in a])

Y = np.array([i[1] for i in a])

X.reshape((27558,1,100,100))

print(X.shape)

print(Y.shape)



Y[0].shape
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train.shape
model = Sequential()

model.add(Conv2D(128,(3,3),input_shape = (100,100,1),strides = 3,activation = "relu"))

# model.add(MaxPooling2D(pool_size = 2))

# model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),strides = 3,activation = "relu"))

# model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),strides = 3,activation = "relu"))

model.add(MaxPooling2D(pool_size = 2))

# model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))

model.add(Dense(2,activation = "softmax"))

model.summary()
model.compile(loss = "categorical_crossentropy",optimizer = "Adam",metrics = ["accuracy"])
X_train = X_train.astype("float32")/255.0

X_test = X_test.astype("float32")/255.0
model.fit(X_train,Y_train,batch_size = 80,epochs = 30,validation_split = 0.18)
acc = model.evaluate(X_test,Y_test)

print("Accuracy : {:.2f}".format(acc[1]*100))
# K.clear_session()



model1 = Sequential()



model1.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(100,100,1)))

model1.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model1.add(MaxPooling2D(pool_size=(2,2)))



model1.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model1.add(MaxPooling2D(pool_size=(2,2)))



model1.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))

model1.add(MaxPooling2D(pool_size=(2,2)))



model1.add(Dropout(0.3))



model1.add(Flatten())

model1.add(Dense(128, activation='relu'))



model1.add(Dropout(0.2))

model1.add(Dense((64)))



model1.add(Dropout(0.2))

model1.add(Dense(2, activation='softmax'))



model1.summary()
import keras

adam =  keras.optimizers.Adam(lr=0.0001)

model1.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(X_train,Y_train, batch_size=80, epochs=30,verbose=1)

acc = model1.evaluate(X_test,Y_test)
print("Accuracy : {:.2f}".format(acc[1]*100))
model3 = Sequential()

model3.add(Conv2D(64,(3,3),strides = 3,input_shape = (100,100,1),padding = "same",activation = "relu"))

model3.add(Conv2D(128,(3,3),strides = 3,padding = "same",activation = "relu"))

model3.add(Conv2D(256,(3,3),strides = 3,padding = "same",activation = "relu"))

model3.add(MaxPooling2D((2,2)))

model3.add(Dropout(0.3))

model3.add(Conv2D(512,(3,3),strides = 3,padding = "same",activation = "relu"))

# model3.add(MaxPooling2D((2,2)))

model3.add(Dropout(0.3))

model3.add(Flatten())

model3.add(Dropout(0.2))

model3.add(Dense(512,activation = "relu"))

model3.add(Dropout(0.2))

model3.add(Dense(256,activation = "relu"))

# model3.add(Dropout(0.2))

model3.add(Dense(64,activation = "relu"))

# model3.add(Dropout(0.2))

model3.add(Dense(2,activation = "softmax"))

model3.summary()













model3.compile(optimizer = "Adam",loss = "categorical_crossentropy",metrics = ["accuracy"])

# model1.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(X_train,Y_train,batch_size = 50,epochs = 50,validation_split = 0.19)
acc3 = model3.evaluate(X_test,Y_test)

print("Model 3 Accuracy : {:.2f}".format(acc3[1]*100))
model4=Sequential()

model4.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(100,100,1)))

model4.add(MaxPooling2D(pool_size=2))

model4.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model4.add(MaxPooling2D(pool_size=2))

model4.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model4.add(MaxPooling2D(pool_size=2))

model4.add(Dropout(0.2))

model4.add(Flatten())

model4.add(Dense(500,activation="relu"))

model4.add(Dropout(0.2))

model4.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model4.summary()

model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model4.fit(X_train,Y_train,batch_size=50,epochs=20,verbose=1,validation_split = 0.18)

acc4 = model4.evaluate(X_test,Y_test)

print("Model 3=4 Accuracy : {:.2f}".format(acc4[1]*100))
CHANNEL_AXIS = 3

stride = 1

def res_layer(x,temp,filters,pooling = False,dropout = 0.0):

    temp = Conv2D(filters,(3,3),strides = stride,padding = "same",kernel_regularizer = keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01))(temp)

    temp = BatchNormalization(axis = CHANNEL_AXIS)(temp)

    temp = Activation("relu")(temp)

    temp = Conv2D(filters,(3,3),strides = stride,padding = "same",kernel_regularizer = keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01))(temp)



    x = add([temp,Conv2D(filters,(3,3),strides = stride,padding = "same",kernel_regularizer = keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01))(x)])

    if pooling:

        x = MaxPooling2D((2,2))(x)

    if dropout != 0.0:

        x = Dropout(dropout)(x)

    temp = BatchNormalization(axis = CHANNEL_AXIS)(x)

    temp = Activation("relu")(temp)

    return x,temp



inp = Input(shape = (100,100,1))

x = inp

x = Conv2D(16,(3,3),strides = stride,padding = "same")(x)

x = BatchNormalization(axis = CHANNEL_AXIS)(x)

x = Activation("relu")(x)

temp = x

x,temp = res_layer(x,temp,32,dropout = 0.2)

x,temp = res_layer(x,temp,32,dropout = 0.4,pooling = True)

x,temp = res_layer(x,temp,64,dropout = 0.2,pooling = True)

x,temp = res_layer(x,temp,128,dropout = 0.2,pooling = True)

x,temp = res_layer(x,temp,256,dropout = 0.4,pooling = True)

x,temp = res_layer(x,temp,256,dropout = 0.4,pooling = True)

x = temp

x = Flatten()(x)

x = Dropout(0.4)(x)

x = Dense(256,activation = "relu")(x)

x = Dropout(0.23)(x)

x = Dense(64,activation = "relu")(x)

x = Dropout(0.2)(x)

x = Dense(2,activation = "softmax")(x)



resnet_model = Model(inp,x,name = "Resnet")

resnet_model.summary()

resnet_model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics = ["accuracy"])

resnet_model.fit(X_train,Y_train,batch_size=120,epochs=30,verbose=1,validation_split = 0.18)

resnet_acc = resnet_model.evaluate(X_test,Y_test)

print("ResNet Accuracy : {:.2f}".format(resnet_acc[1]*100))