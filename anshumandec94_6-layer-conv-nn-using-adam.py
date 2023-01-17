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



import tensorflow as tf





# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/Kannada-MNIST/train.csv")

test_df=pd.read_csv("../input/Kannada-MNIST/test.csv")



x_train=train_df.drop(["label"], axis=1).values.astype('float32')

y_train=train_df["label"].values.astype("int32")



x_test=test_df.drop(["id"],axis=1).values.astype('float32')

test_df.columns
x_train=x_train.reshape(x_train.shape[0], 28, 28)/255.0

x_test=x_test.reshape(x_test.shape[0], 28, 28)/255.0
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)

print(x_test.shape)
x_train=x_train.reshape(x_train.shape[0], 28,28,1)

x_test=x_test.reshape(x_test.shape[0],28,28,1)

x_val=x_val.reshape(x_val.shape[0],28,28,1)
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Activation, ReLU, Flatten, Dropout, BatchNormalization

from keras.models import Model



X=Input(shape=[28,28,1])

x=Conv2D(16, (3,3), strides=1, padding="same", name="conv1")(X)

x=BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform", name="batch1")(x)

x=Activation('relu',name='relu1')(x)

x=Dropout(0.1)(x)



x=Conv2D(32, (3,3), strides=1, padding="same", name="conv2")(x)

x=BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch2")(x)

x=Activation('relu',name='relu2')(x)

x=Dropout(0.15)(x)

x=MaxPooling2D(pool_size=2, strides=2, padding="same", name="max2")(x)



x=Conv2D(64, (5,5), strides=1, padding ="same", name="conv3")(x)

x=BatchNormalization(momentum=0.17, epsilon=1e-5, gamma_initializer="uniform", name="batch3")(x)

x=Activation('relu', name="relu3")(x)

x=MaxPooling2D(pool_size=2, strides=2, padding="same", name="max3")(x)



x=Conv2D(128, (5,5), strides=1, padding="same", name="conv4")(x)

x=BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch4")(x)

x=Activation('relu', name="relu4")(x)

x=Dropout(0.17)(x)



x=Conv2D(64, (3,3), strides=1, padding="same", name="conv5")(x)

x=BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch5")(x)

x=Activation('relu', name='relu5')(x)

#x=MaxPooling2D(pool_size=2, strides=2, padding="same", name="max5")(x)

x=Dropout(0.2)(x)



x=Conv2D(32, (3,3), strides=1, padding="same", name="conv6")(x)

x=BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform", name="batch6" )(x)

x=Activation('relu', name="relu6")(x)

x=Dropout(0.05)(x)

#x=MaxPooling2D(pool_size=2, strides=2, padding="same", name="max6")(x)



x=Flatten()(x)

x=Dense(50, name="Dense1")(x)

x=Activation('relu', name='relu7')(x)

x=Dropout(0.05)(x)

x=Dense(25, name="Dense2")(x)

x=Activation('relu', name='relu8')(x)

x=Dropout(0.03)(x)

x=Dense(10, name="Dense3")(x)

x=Activation('softmax')(x)



model=Model(inputs=X, outputs=x)

print(model.summary())
from keras.preprocessing. image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam



checkpoint=ModelCheckpoint('bestweights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode="max")



epochs=80

lr=0.002

optimizer=Adam(lr=lr, decay=lr/(epochs*1.5))



model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",metrics=["accuracy"])



datagenerator=ImageDataGenerator( rotation_range=9, zoom_range=0.25, width_shift_range=0.25, height_shift_range=0.25)



datagenerator.fit(x_train)

batch_size=64

history=model.fit_generator(datagenerator.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_val, y_val), verbose=2,steps_per_epoch=x_train.shape[0]//batch_size, callbacks=[checkpoint])

model.load_weights("bestweights.hdf5")
results=model.predict(x_test)
results=np.argmax(results, axis=1)
results
dg=pd.DataFrame()

dg['id']=list(test_df.values[0:,0])
dg['label']=results
dg
dg.to_csv("submission.csv", index=False)