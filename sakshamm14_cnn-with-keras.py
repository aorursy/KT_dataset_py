import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,BatchNormalization,Dropout,Flatten
from keras import activations,optimizers 
from keras.datasets import fashion_mnist
train=pd.read_csv("../input/fashion-mnist_train.csv")
test=pd.read_csv("../input/fashion-mnist_test.csv")
x_train=train.iloc[:,1:]
y_train=train.iloc[:,0]
x_test=test.iloc[:,1:]
y_test=test.iloc[:,0]
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
x_train.shape
import matplotlib.pyplot as plt
import keras.utils




x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


y_test=y_test.reshape(y_test.shape[0],1)
y_train=y_train.reshape(y_train.shape[0],1)

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
y_train=encoder.fit_transform(y_train)
y_test=encoder.transform(y_test)

x_train1=x_train[:10000]
y_train1=y_train[:10000]
x_test1=x_test[:1000]
y_test1=y_test[:1000]
y_train1=y_train1.todense()
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
op=optimizers.Adam(lr=0.0001)
model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=15)
ypred=model.predict(x_test)
ypred=np.argmax(ypred,axis=-1)
ytests1=np.argmax(y_test,axis=-1)
ytests1=np.array(ytests1)
ytests1=ytests1.reshape(ytests1.shape[0])
(ypred==ytests1).sum()







