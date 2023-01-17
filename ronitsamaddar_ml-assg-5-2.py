# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


imgSet = np.load('../input/imgSet.npy')
ySet = np.load('../input/ySet.npy')
print(ySet.shape)
print(imgSet.shape)
print(np.unique(ySet))

shuffle = np.random.permutation(ySet.shape[0])
X = imgSet[shuffle]
y = ySet[shuffle]
print(y)
y[y=='daisy'] = 0
y[y=='dandelion'] = 1
y[y=='rose'] = 2
y[y=='sunflower'] = 3
y[y=='tulip'] = 4
y_enc = np.zeros([y.shape[0],5])
y = y.astype('int')
for i,c in enumerate(y):
    for j in range(5):
        #print(c==j)
        if c==j:
            y_enc[i,j] = 1
            #print("Hello")
    
print(y_enc[40:50,:])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_enc,train_size=0.9)
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,Activation,BatchNormalization ,Flatten,MaxPool2D
model = Sequential()
model.add(Conv2D(32,(3,3),padding='valid',activation='relu',input_shape=(120,120,3)))
model.add(Conv2D(64,(3,3),padding='valid',strides=2,activation='relu'))
model.add(Conv2D(512,(3,3),padding='valid',strides=4,activation='relu'))
model.add(MaxPool2D(2,strides=2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(X_train,y_train,epochs=10,batch_size=128,verbose=1,validation_split=0.1)
res_rgb_train=model.evaluate(X_train,y_train)
print(res_rgb_train)
res_rgb_test=model.evaluate(X_test,y_test)
print(res_rgb_test)
model.save('ass5.h5')
from skimage import color
X_train_gr = np.zeros([X_train.shape[0],120,120])
X_test_gr = np.zeros([X_test.shape[0],120,120])
for i in range(X_train.shape[0]):
    X_train_gr[i] = color.rgb2gray(X_train[i])
for i in range(X_test.shape[0]):
    X_test_gr[i] = color.rgb2gray(X_test[i])
model_gr = Sequential()
model_gr.add(Conv2D(32,(3,3),padding='valid',activation='relu',input_shape=(120,120,1)))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(Conv2D(64,(3,3),padding='valid',strides=2,activation='relu'))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(Conv2D(128,(3,3),padding='valid',strides=4,activation='relu'))
model_gr.add(BatchNormalization(axis=-1))
model_gr.add(MaxPool2D(2))
model_gr.add(Flatten())
model_gr.add(Dense(256,activation='relu'))
model_gr.add(Dense(64,activation='relu'))
model_gr.add(Dense(5,activation='softmax'))
model_gr.summary()
model_gr.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist_gr = model_gr.fit(X_train_gr.reshape(-1,120,120,1),y_train,epochs=10,batch_size=64,verbose=1,validation_split=0.1)
res_gr_train=model_gr.evaluate(X_train_gr.reshape(-1,120,120,1),y_train)
print(res_gr_train)
res_gr_test=model_gr.evaluate(X_test_gr.reshape(-1,120,120,1),y_test)
print(res_gr_test)
model.save('ass5_gr.h5')
plt.subplot(121)
names=['RGB','GRAYSCALE']
plt.title('Training Accuracy')
plt.bar(names,height=[res_rgb_train[1], res_gr_train[1]],color=['r','y'])
plt.subplot(122)
plt.title('Test Accuracy')
plt.bar(names,height=[res_rgb_test[1], res_gr_test[1]],color=['r','y'])
plt.show()
Flowers=["daisy","dandelion","rose","sunflower","tulip"]
for i in range(5):
    plt.imshow(X_test[i])
    pred = np.argmax(model.predict(X_test[i].reshape(1,120,120,3)))
    plt.title('Flower = '+str(pred)+"("+Flowers[pred]+")")
    plt.show()
Flowers=["daisy","dandelion","rose","sunflower","tulip"]
for i in range(5):
    gray=color.rgb2gray(X_test[i])
    plt.imshow(gray)
    pred = np.argmax(model_gr.predict(gray.reshape(1,120,120,1)))
    plt.title('Flower = '+str(pred)+"("+Flowers[pred]+")")
    plt.show()
loss_rgb = hist.history['loss']
epoch_rgb = list(range(len(loss_rgb)))
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
