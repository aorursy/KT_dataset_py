# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical

import keras

from keras import callbacks

from keras.layers import Conv2D,Dense,MaxPooling2D,Dropout,Flatten,BatchNormalization

from keras.models import Sequential

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Loding Data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train_y = train["label"]

train_x = train.drop("label",axis = 1)
train_x.shape,train_y.shape
g = sns.countplot(train_y)

train_y = to_categorical(train_y)
train_x = train_x.values.reshape(len(train_x),28,28,-1)

test = test.values.reshape(len(test),28,28,-1)
train_x.shape
model = Sequential()
model.add(Conv2D(32,(3,3), strides=(1, 1), padding='same', activation="relu",input_shape = (28,28,1),data_format = "channels_last", use_bias = True))

model.add(Conv2D(32,(3,3), strides=(1, 1), padding='same', activation="relu", use_bias = True))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3), strides=(1, 1), padding='same', activation="relu", use_bias = True))

model.add(Conv2D(64,(3,3), strides=(1, 1), padding='same', activation="relu", use_bias = True))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256,activation = "relu", use_bias = True))

model.add(Dropout(0.5))

model.add(Dense(10,activation = "softmax",use_bias = True))

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer,loss = "categorical_crossentropy",metrics = ['accuracy'])
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='loss',patience=3, verbose=1,factor=0.2,min_lr=0.00001)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(train_x)
model.fit_generator(datagen.flow(train_x,train_y,batch_size = 100),epochs = 30,steps_per_epoch=train_x.shape[0] // 100, callbacks=[learning_rate_reduction])
y_pred = model.predict(test)
y_pred
y_pred = np.array(y_pred)
y_pred
y_pred_final = []

for i in y_pred:

    y_pred_final.append(np.argmax(i))

    

    
results = pd.Series(y_pred_final,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit_mnist.csv",index=False)