#importing all the required libraries

import numpy as np#numpy array calculations

import pandas as pd#working with dataframe

import matplotlib.pyplot as plt#for visualizing the plots

import matplotlib.image as mpimg#To view images which are of the form of numbers

import seaborn as sns#For plotting



np.random.seed(0)#To get same results whenever i do this

from sklearn.model_selection import train_test_split#For validation and checking the preformance of model

from sklearn.metrics import confusion_matrix#To see where our model doing wrong

import itertools#For efficient looping

from keras.utils.np_utils import to_categorical#one hot encoding

from keras.models import Sequential#The CNN type it have other types like Residual etc.,.

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D#Types of layers

from keras.optimizers import RMSprop,Adam#fits the filter variables,weights etc.,.

from keras.preprocessing.image import ImageDataGenerator#For data augmentation

from keras.callbacks import ReduceLROnPlateau#To make sure the we reduce learning rate if model stopped learning, upto some limit
sns.set(style='white',context='notebook',palette='deep')
#Load the data

train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
y_train=train['label']

X_train=train.drop(labels=['label'],axis=1)

del train
test.head()
X_train.describe()
sns.countplot(y_train)
X_train.isnull().sum()
X_train=X_train/255.0

test=test/255.0
X_train=X_train.values.reshape(-1,28,28,1)#making an 3D array like (28*28*1) how an actual image look.

test=test.values.reshape(-1,28,28,1)
y_train=to_categorical(y_train,num_classes=10)

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=0)#specify random state to get same outputs
g=plt.imshow(X_train[1][:,:,0])#displays image


model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(10,activation='softmax'))

optimizer=Adam()
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
learning_rat=ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.001)

epochs=3

batch_size=64 #powers of two is better generally and by convention we follow this
datagenerated=ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)
datagenerated.fit(X_train)
digit_rec=model.fit_generator(datagenerated.flow(X_train,y_train,batch_size=batch_size),

                              epochs=epochs,validation_data=(X_val,y_val),verbose=2,steps_per_epoch=X_train.shape[0],

                              callbacks=[learning_rat])
results=model.predict(test)

results=np.argmax(results,axis=1)

results=pd.Series(results,name='Label')
submission=pd.concat([pd.Series(range(1,28001),name='ImageId'),results],axis=1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)