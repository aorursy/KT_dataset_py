import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout
from keras.layers import Dense,Activation,Flatten
from keras.utils import multi_gpu_model
from keras import backend as K
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import argmax
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

data=pd.read_csv("../input/hw2q3train.csv")    #training dataset using pandas 
test_data=pd.read_csv("../input/hw2q3test.csv")  #test dataset " """"""""""
train=np.array(data)        #convert training dataset into numpy array from the pandas dataframe
train.shape
x_train=train[:,1:]         #training features
y_train=train[:,0]          #training labels
test=np.array(test_data)      #convert test dataset into numpy array from pandas dataframe
test.shape
x_test=test[:,1:]         #test features values
y_test=test[:,0]          #test labels
print("no of training examples:",x_train.shape[0])
print("no of test examples:",x_test.shape[0])
length=x_train.shape[0]
xx_train=[]
for i in range(length):
    t=x_train[i,:]
    t=t.reshape(28,28)
    xx_train.append(t)
xx_train=np.array(xx_train)
length=x_test.shape[0]
#print(length)
xx_test=[]
for i in range(length):
    t=x_test[i,:]
    t=t.reshape(28,28)
    xx_test.append(t)
xx_test=np.array(xx_test)
print(xx_train.shape)
print(xx_test.shape)

#plt.imshow(xx_train[7])  #display the random image from training dataset
plt.imshow(xx_train[199])
xx_train=xx_train/255
xx_test=xx_test/255
xx_train.shape
xx_train,x_valid=xx_train[:50000],xx_train[50000:]
y_train,y_valid=y_train[:50000],y_train[50000:]
print("no of traiing set:",xx_train.shape[0])
print(' no of valid set:',x_valid.shape[0])
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)

y_test = keras.utils.to_categorical(y_test, 10)
xx_train=xx_train.reshape(xx_train.shape[0],28,28,1)
x_valid=x_valid.reshape(x_valid.shape[0],28,28,1)
xx_test=xx_test.reshape(xx_test.shape[0],28,28,1)
model=Sequential()
model.add(Conv2D(64,kernel_size=2,strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.3))
model.add(Conv2D(32,kernel_size=2,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(1))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
score=model.fit(xx_train,y_train,epochs=10,batch_size=256,validation_data=[x_valid,y_valid])
history=model.evaluate(xx_test,y_test,batch_size=256)
print("test_accuracy:",history[1])        #accuracy on test dataset
y_pred=model.predict(xx_test)    
y_test.shape
y_pred=argmax(y_pred,axis=1)
y_test=argmax(y_test,axis=1)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
matrix=confusion_matrix(y_test,y_pred,labels=[0,1,2,3,4,5,6,7,8,9])
matrix
y_pred[6]

y_test[6]
unlabel_data=pd.read_csv("../input/hw2q3unlabeled.csv")
unlabel=np.array(unlabel_data)
unlabel.shape
x_unlabels=unlabel[:,1:]
length=x_unlabels.shape[0]
x_labels=[]
for i in range(length):
    t=x_unlabels[i,:]
    t=t.reshape(28,28)
    x_labels.append(t)
x_labels=np.array(x_labels)
x_labels=x_labels.reshape(x_labels.shape[0],28,28,1)
y_labels=model.predict(x_labels)
y_labels=argmax(y_labels,axis=1)
pd.DataFrame(y_labels).to_csv("o_p.csv",index=False)
t=pd.read_csv('o_p.csv')
t
pd.DataFrame(matrix).to_csv("cf.csv",index=False)
m=pd.read_csv('cf.csv')
m
