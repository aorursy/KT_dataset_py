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
import os

print(os.listdir("../input"))
import pandas as pd

from pandas import DataFrame,Series

import numpy as np
train=pd.read_csv("../input/train.csv",sep=',')
test=pd.read_csv("../input/test.csv",sep=',')
train.head()
test.head()
train.shape
test.shape
x_train=train.drop('label',axis=1)
y_train=train['label']
x_train=x_train/255.0
test=test/255.0
y_train.unique()
import keras

import matplotlib.pyplot as plt

%matplotlib inline
plt.imshow(np.array(x_train.iloc[0]).reshape((28,28)),cmap='gray')
import sklearn.model_selection as model_selection

x_train,x_test,y_train,y_test=model_selection.train_test_split(x_train,y_train,test_size=0.20,random_state=200)
from keras.models import Sequential

from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense,Flatten

from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD

from keras.layers import Dropout
model=Sequential()

model.add(Conv2D(filters=6,kernel_size=(3,3),padding='same',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.2,seed=100))

model.add(Dense(120,activation='relu'))

model.add(Dense(84,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.summary()
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
x=np.array(x_train)

x_train=x.reshape(x.shape[0],28,28,1)

y=keras.utils.to_categorical(np.array(y_train),10)

m=model.fit(x_train,y,epochs=10,batch_size=1000,validation_split=0.20)
x_test=np.array(x_test)

x_test=x_test.reshape(x_test.shape[0],28,28,1)
plt.imshow(x_test[8,:,:].reshape(28,28),cmap='gray')
p=model.predict_proba(x_test[8,:,:].reshape(-1,28,28,1))
np.argmax(p)
results=model.predict(x_test)
results=np.argmax(results,axis=1)
results=pd.Series(results,name='label')
submission=pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("kaggle_digitrecognizer_datasolved.csv",index=False)