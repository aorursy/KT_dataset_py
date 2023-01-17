# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x=pd.read_csv('../input/train.csv')
y=pd.read_csv('../input/test.csv')
z=x['label']
x=x.drop(['label'],axis=1)
x=x.applymap(lambda f :f/255)
y=y.applymap(lambda f :f/255)
x=pd.DataFrame(x)
y=pd.DataFrame(y)
x=x.values.reshape(-1,28,28,1)
y=y.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical
z=to_categorical(z,num_classes=10)
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,AvgPool2D
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ztrain,ztest=tts(x,z,train_size=0.8)
from keras.models import Sequential
from keras.layers import MaxPool2D,AvgPool2D
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))
from keras.preprocessing.image import ImageDataGenerator
data=ImageDataGenerator(height_shift_range=0.1,width_shift_range=0.1,rotation_range=0.1,zoom_range=0.1)
from keras.callbacks import ReduceLROnPlateau 
lrs=ReduceLROnPlateau(patience=2,factor=0.5,monitor='val_acc')

model.compile(loss='categorical_crossentropy',optimizer='RMSProp',metrics=['accuracy'])

history=model.fit_generator(data.flow(xtrain,ztrain,batch_size=20),steps_per_epoch=500,epochs=44,samples_per_epoch=xtrain.shape[0]//50,callbacks=[lrs],validation_data=[xtest,ztest])
plt.subplot(2,1,1)
plt.plot(history.history['loss'],c='blue')
plt.plot(history.history['val_loss'],c='red')
plt.subplot(2,1,2)
plt.plot(history.history['acc'],c='blue')
plt.plot(history.history['val_acc'],c='red')
c=model.predict(y)
result=np.argmax(c,axis=1)
s=pd.DataFrame(result,index=np.arange(1,28001))
s.columns=['Label']
s.index.name='ImageId'
s.to_csv('result.csv')
