import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Activation,Dropout,Dense,Conv2D,AveragePooling2D,Flatten,ZeroPadding2D,MaxPooling2D

from keras import optimizers

from sklearn.model_selection import train_test_split

from keras.models import Sequential

import seaborn as sns

from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical 

import math
test=pd.read_csv('../input/test.csv',delimiter=',')

train=pd.read_csv('../input/train.csv',delimiter=',')

print(train.head())

label=train['label']

print(label.shape)

del train['label']

print(label.head())

sns.countplot(label)

train=train.values

train=train.reshape(train.shape[0],28,28,1)

test=test.values

test=test.reshape(test.shape[0],28,28,1)

plt.figure(figsize=(25,10))

for i in range(0,10):

    plt.subplot(1,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(train[i][:,:,0],cmap='gray')

    #these are the sample images  
label=to_categorical(label,10)

X_train, X_test, Y_train, Y_test = train_test_split(train,label, test_size = 0.1)
model=Sequential()

def build():

    model.add(Conv2D(6,(5,5),strides=1,padding='valid', input_shape = (28,28,1)))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    model.add(Conv2D(16,(5,5),strides=1,padding='valid'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(120,activation='relu'))

    model.add(Dense(84,activation='relu'))

    model.add(Dense(10,activation='softmax'))

    model.compile(optimizer='AdaDelta',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

    model.fit(X_train,Y_train,epochs=30,batch_size=128,verbose=1)

    

build()
modelsub=model.predict(X_train,batch_size=None, verbose=1)

c=0

f=0

for i in range(modelsub.shape[0]):

    if np.argmax(modelsub[i])==np.argmax(Y_train[i]):

        c+=1

    else:

        f+=1

accuracy=c/(c+f)

print('Train Accuracy:',accuracy*100)

a=model.evaluate(X_test, Y_test,verbose=1)

print('Test Accuracy',a[1])
mode=model.predict(test,batch_size=None, verbose=1)

ls=[0,0,0,0,0,0,0,0,0,0]

for i in range(modelsub.shape[0]):

    ls[np.argmax(modelsub[i])]+=1

print(ls)

count1=0

count2=0

for i in range(modelsub.shape[0]):

    if(np.argmax(Y_train[i])==np.argmax(modelsub[i])):

        count1+=1

    else:

        count2+=1

print(count1,' ',count2)
count=0

plt.figure(figsize=(25,10))

while count<10:

    for i in range(modelsub.shape[0]):

        if(np.argmax(Y_train[i])!=np.argmax(modelsub[i])):

            plt.subplot(1,10,count+1)

            plt.xticks([])

            plt.yticks([])

            s='Predicted:'

            s=s+' '+str(np.argmax(modelsub[i]))

            plt.xlabel(s)

            plt.imshow(train[i][:,:,0],cmap='gray')

            count+=1

            if count is 10:

                break

            
ccc=[]

for i in range(mode.shape[0]):

    ccc.append(np.argmax(mode[i]))

d=np.arange(0,test.shape[0])+1

d.shape

df=pd.DataFrame({'ImageId':d,'Label':ccc

},index=d)

df.to_csv("cnn_mnist_datagen.csv",index=False)
