# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X = np.array(pd.read_csv('../input/train.csv'))
Y = X[:,0]
X = X[:,1:]
X = X.astype(np.float32)/255.0
Y = to_categorical(Y,num_classes=10)
X = np.reshape(X,(-1,28,28,1))
fig = plt.figure(figsize=(15,15))
for i in range(5):
    fig.add_subplot(1,5,i+1)
    plt.imshow(X[i,:,:,0])
plt.show()
#Splitting into train and validation data
# SPLIT_RATIO = 0.02
# TOTAL = X.shape[0]
# TRAIN = int(float(TOTAL)*(1-SPLIT_RATIO))
# x_train = X[:TRAIN,:,:,:]
# x_val = X[:TRAIN,:,:,:]
# y_train = Y[TRAIN:]
# y_val = Y[TRAIN:]
x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=0.05)
print(f'Shape of training data set is {x_train.shape}')
print(f'Shape of validation dataset is {x_val.shape}')
print(f'Shape of training outputs is {y_train.shape}')
print(f'Shape of validation outputs is {y_val.shape}')
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop,Adadelta,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
#Implement data augmentation
datagen = ImageDataGenerator(rotation_range=2.5)
datagen.fit(x_train)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=16,kernel_size=(1,1),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(1,1),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(10,activation="softmax"))
print(model.summary())
optimizer = RMSprop()
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
earlystopping = EarlyStopping(patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor="val_acc",patience=3,verbose=1,factor=0.5,min_lr=0.00001)
checkpoint = ModelCheckpoint('weights.hdf5',save_best_only=True,monitor='val_acc',verbose=1)
epochs = 200
batch_size = 128
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),epochs=epochs,validation_data=(x_val,y_val),verbose=2,callbacks=[learning_rate_reduction,checkpoint])
model.load_weights('weights.hdf5')
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
#load test data
X_test = np.array(pd.read_csv('../input/test.csv'))
X_test = X_test.astype(np.float32)/255.0
X_test = np.reshape(X_test,(-1,28,28,1))
print(X_test.shape)
predictions = np.argmax(model.predict(X_test),axis=1)
print(predictions)
predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name="ImageId"),predictions],axis=1)
submission.to_csv("cnn_mnist_submission.csv",index=False)