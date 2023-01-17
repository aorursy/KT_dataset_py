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
XNumpy=np.load("../input/imgSet.npy")
YNumpy=np.load("../input/ySet.npy")
print(XNumpy.shape)
#print(YNumpy.shape)
#Shuffling all rows

length=len(XNumpy)
arr=np.arange(length)
import random
np.random.shuffle(arr)
print(arr)
print(len(arr))





X=XNumpy[arr]
Y=YNumpy[arr]
Y[Y=='daisy']=0
Y[Y=='dandelion']=1
Y[Y=='rose']=2
Y[Y=='sunflower']=3
Y[Y=='tulip']=4
Y=Y.astype('int')
print(Y)
#Now do one-hot encoding
Y_OneHotEnc=[[0 for x in range(5)]for y in range(length)]
print(Y_OneHotEnc[100][4])
for i in range(0,length):
    for j in range(0,5):
        if(Y[i]==j):
            Y_OneHotEnc[i][j]=1
print(Y_OneHotEnc[0])
print(Y_OneHotEnc[1])

Y_OHC=np.array(Y_OneHotEnc)
print(Y_OHC.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y_OHC,train_size=0.9)
print("X Training set :"+str(X_train.shape))
print("Y Training set :"+str(Y_train.shape))
print("X Test set :"+str(X_test.shape))
print("Y Test set :"+str(Y_test.shape))

from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,Activation,BatchNormalization ,Flatten,MaxPool2D

model=Sequential()
model.add(Conv2D(8,(3,3), padding='same',activation='relu',input_shape=(120,120,3)))
model.add(MaxPool2D(2))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))

model.add(Conv2D(16,(3,3) , padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))


model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1,validation_split=0.1)


model.save('ass5.h5')
#Grayscale images 
#Convert all the 3 d feature vectors to 2 d feature vectors in training and test set

from skimage import color
Grayscale_X_train=np.zeros([X_train.shape[0],120,120])
Grayscale_X_test=np.zeros([X_test.shape[0],120,120])
for i in range(0,X_train.shape[0]):
    Grayscale_X_train[i]=color.rgb2gray(X_train[i])

for i in range(0,X_test.shape[0]):
    Grayscale_X_test[i]=color.rgb2gray(X_test[i])
print(Grayscale_X_train[0].shape)
print(Grayscale_X_test[0].shape)

model_gr = Sequential()
model_gr.add(Conv2D(6,(3,3),padding='valid',activation='relu',input_shape=(120,120,1)))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(Conv2D(8,(3,3),padding='valid',activation='relu'))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(Conv2D(10,(3,3),padding='valid',activation='relu'))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(MaxPool2D(2))
model_gr.add(Flatten())
model_gr.add(Dense(256,activation='relu'))
model_gr.add(Dense(64,activation='relu'))
model_gr.add(Dense(5,activation='softmax'))
model.summary()
model_gr.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist_gr = model_gr.fit(Grayscale_X_train.reshape(-1,120,120,1),Y_train,epochs=10,batch_size=64,verbose=1,validation_split=0.1)
res_gr_train=model_gr.evaluate(Grayscale_X_train.reshape(-1,120,120,1),Y_train)
print(res_gr_train)

res_gr_test=model_gr.evaluate(Grayscale_X_test.reshape(-1,120,120,1),Y_test)
print(res_gr_test)
res_rgb_train=model.evaluate(X_train,Y_train)
print(res_rgb_train)
res_rgb_test=model.evaluate(X_test,Y_test)
print(res_rgb_test)
#Accuracy comparison

import matplotlib.pyplot as plt
plt.subplot(121)
names=['RGB','GRAYSCALE']
plt.title('Training Accuracy')
plt.bar(names,height=[res_rgb_train[1], res_gr_train[1]],color=['g','r'])
plt.subplot(122)
plt.title('Test Accuracy')
plt.bar(names,height=[res_rgb_test[1], res_gr_test[1]],color=['g','r'])
plt.show()



list1=["tulip","rose","dandelion","daisy","sunflower"]

for i in range(0,5):
    plt.imshow(X_test[i])
    pred=np.argmax(model.predict(X_test[i].reshape(1,120,120,3)))
    plt.title('Flower = '+str(pred)+"("+list1[pred]+")")
    plt.show()


for i in range(0,5):
    x=color.rgb2gray(X_test[i])
    plt.imshow(x)
    pred=np.argmax(model_gr.predict(x.reshape(1,120,120,1)))
    plt.title('Flower = '+str(pred)+"("+list1[pred]+")")
    plt.show()
#Epoch wise training accuracy and loss
loss_rgb = hist.history['loss']
epoch_rgb = list(range(len(loss_rgb)))
print("Epoch rgb: "+str(epoch_rgb))
loss_gr = hist_gr.history['loss']

plt.ylabel('loss ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,loss_rgb,label='rgb model')
plt.plot(epoch_rgb,loss_gr,label='gray model')
plt.legend()
plt.show()
acc_rgb = hist.history['acc']
epoch_rgb = list(range(len(loss_rgb)))
acc_gr = hist_gr.history['acc']

plt.ylabel('accuracy ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,acc_rgb,label='rgb model')
plt.plot(epoch_rgb,acc_gr,label='gray model')
plt.legend()
plt.show()
loss_rgb = hist.history['val_acc']
epoch_rgb = list(range(len(loss_rgb)))
loss_gr = hist_gr.history['val_acc']

plt.ylabel('validation accuracy ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,loss_rgb,label='rgb model')
plt.plot(epoch_rgb,loss_gr,label='gray model')
plt.legend()
plt.show()
loss_rgb = hist.history['val_loss']
epoch_rgb = list(range(len(loss_rgb)))
loss_gr = hist_gr.history['val_loss']

plt.ylabel('validation loss ')
plt.xlabel('epoch')
plt.plot(epoch_rgb,loss_rgb,label='rgb model')
plt.plot(epoch_rgb,loss_gr,label='gray model')
plt.legend()
plt.show()
