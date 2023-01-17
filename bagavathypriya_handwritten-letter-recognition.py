# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv").astype('float32')

df.head()
df.tail()
df.rename(columns={'0':'label'}, inplace=True)
df.info()
df.describe()
df.head()
x=df.drop('label',axis=1)
y=df['label']
def convert(text):
    dic = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
                        11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
    return dic[text]

data=df.copy()
data['mapped']=data['label'].apply(lambda df:convert(df))
type(y)
type(x)
print(x.shape)
print(y.shape)
from sklearn.utils import shuffle
x=shuffle(x)
a=x.iloc[44,:]
print(a.shape)
b=a.values.reshape(28,28)
print(b.shape)
plt.imshow(b,cmap='gray')
plt.title('Image of a random alphabet')
abc=data['mapped']
sns.countplot(abc)
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
scalar=MinMaxScaler()
scalar.fit(xtrain)
xtrain=scalar.transform(xtrain)
xtest=scalar.transform(xtest)
xshuff=shuffle(xtrain)
a=xshuff[46,:]
a.shape
b=a.reshape(28,28)
b.shape
plt.imshow(b,cmap='gray')
plt.title('Image after scaling',fontsize=20)
xtrain=xtrain.reshape(xtrain.shape[0],28,28,1).astype('float32')
xtest=xtest.reshape(xtest.shape[0],28,28,1).astype('float32')
from keras.utils import to_categorical
from numpy import array
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
model=Sequential()
model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))           #Convolutional layer
model.add(MaxPooling2D(pool_size=(2,2)))                                      #Max Pooling
model.add(Dropout(0.3))                                                        #Dropout
model.add(Flatten())                                                          #Flattening the layer
model.add(Dense(128,activation='relu'))                                       #First layer of NN
model.add(Dense(len(y.unique()),activation='softmax'))                        #Final layer of NN
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
his=model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=18,batch_size=200,verbose=2)
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))           #Convolutional layer
model.add(MaxPooling2D(pool_size=(2,2)))                                      #Max Pooling
model.add(Conv2D(64,(3,3),activation='relu'))           #Second Convolutional layer
model.add(MaxPooling2D(pool_size=(2,2)))    
model.add(Conv2D(128,(3,3),activation='relu'))           #Third Convolutional layer
model.add(MaxPooling2D(pool_size=(2,2)))    
model.add(Dropout(0.3))                                                        #Dropout
model.add(Flatten())                                                          #Flattening the layer
model.add(Dense(128,activation='relu'))                                       #First layer of NN
model.add(Dense(len(y.unique()),activation='softmax'))                        #Final layer of NN
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist=model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=30,batch_size=200,verbose=2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
ypred=model.predict(xtest)
print(ypred)
from sklearn import metrics
from sklearn.metrics import confusion_matrix

err=metrics.mean_squared_error(ytest,ypred)
print(err)
fig=xtrain[650,:]
fig=fig.reshape(28,28)
plt.imshow(fig,cmap='gray')
print(ypred[650])
