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
from keras.layers.core import Dense,Activation,Dropout

from keras import Sequential

import numpy as np



import pandas as pd
df=pd.read_csv('../input/A_Z Handwritten Data/A_Z Handwritten Data.csv')

df.shape
seed=784

X = df.iloc[:,0:784]

Y=df.iloc[:,0]

X.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=seed)
print(x_train.values[0].shape)
#x_train =  x_train.values.reshape(x_train.shape[0],)

#x_test  =  x_test.values.reshape(x_test.shape[0],784)

x_train =  x_train.astype('float32')

x_test  =  x_test.astype('float32')

x_train /=255

x_test/=255

print(x_train.shape[0],'train samples')

print(x_test.shape[0],'test samples')
nb_classes=26 # from a to z

y_train.shape
from keras.utils import np_utils

print(y_train.shape)

y_train = np_utils.to_categorical(y_train,num_classes=26)

y_test = np_utils.to_categorical(y_test,num_classes=26)
model=Sequential()

model.add(Dense(512,input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(26))

model.add(Activation('softmax'))

model.summary()
import matplotlib.pyplot as plt

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(x_train.values[i].reshape(28,28),cmap='gray')
from keras.optimizers import SGD,Adam,RMSprop,Adamax

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
batch_size=20

nb_epoch=20



model.fit(x_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test,y_test))

score, acc = model.evaluate(x_test,y_test,batch_size=batch_size)

print('Test Score:',score)

print('Test Accuracy:',acc*100)