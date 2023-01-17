# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,BatchNormalization
from keras.optimizers import Adam,Adagrad #optimizers
from keras.datasets import mnist #Import datasets from inbulit Keras Datasets

# Any results you write to the current directory are saved as output.

# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

s=np.array(train)
labels=s[:,0]
pixels=s[:,1:785]
labels=keras.utils.to_categorical(labels,10)
#Data Reshaping
pixels=pixels.reshape(42000,28,28,1);
pixels=pixels/255
for i in range(10):
    print("label is:",labels[i])
    plt.imshow(pixels[i])
    plt.show()




from sklearn.model_selection import train_test_split
X_train,X_test,label_train,label_test=train_test_split(pixels,labels,test_size=0.2,random_state=0)
model=Sequential()
model.add(Conv2D(64,kernel_size=(3,3),padding='Same',activation='relu',input_shape=(28,28,1,)))
model.add(Conv2D(32,kernel_size=(2,2),padding='Same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(Conv2D(32,kernel_size=(2,2),padding='Same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(24,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(Conv2D(12,kernel_size=(2,2),padding='Same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer=Adam(),metrics=['accuracy'])
model.fit(X_train,label_train,batch_size=64,epochs=40,validation_data=(X_test,label_test))
for i in range(20):
    print(model.predict(pixels[i].reshape(1,28,28,1)).argmax())
    plt.imshow(pixels[i].reshape(28,28))
    plt.show()
    
s1=np.array(test)
pixels1=s1[:,0:784]
#Data Reshaping
pixels1=pixels1.reshape(28000,28,28,1);
pixels1=pixels1/255
mad=[0]*28000
sid=[0]*28000
for i in range(28000):
    #print(model.predict(pixels1[i].reshape(1,28,28,1)).argmax()," ",i+1)
    mad[i]=model.predict(pixels1[i].reshape(1,28,28,1)).argmax()
    sid[i]=i+1

np.array(mad)
np.array(sid)



my_sub=pd.DataFrame({"ImageId":sid,"Label":mad})
my_sub.to_csv('submission.csv', index=False)