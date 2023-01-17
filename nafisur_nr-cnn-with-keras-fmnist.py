import keras
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_train=pd.read_csv('../input/fashion-mnist_train.csv')
df_test=pd.read_csv('../input/fashion-mnist_test.csv')
print(df_train.shape)
print(df_test.shape)
df_train.head()
df_test.head()
X_train=df_train.iloc[:,1:].values
y_train=df_train.iloc[:,0].values
X_test=df_test.iloc[:,1:].values
y_test=df_test.iloc[:,0].values
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train=X_train.astype(float)
X_test=X_test.astype(float)
X_train /=255
X_test /=255
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.compile(loss=categorical_crossentropy,optimizer=SGD(lr=0.05,momentum=0.9,decay=1e-6),metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_test,y_test))
