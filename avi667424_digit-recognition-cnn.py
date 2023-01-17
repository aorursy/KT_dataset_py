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
x_train=pd.read_csv('../input/train.csv')
x_test_data=pd.read_csv('../input/test.csv')
x_train.shape, x_test_data.shape
from sklearn.model_selection import train_test_split
x=x_train.iloc[:,1:785]

y=x_train.iloc[:,0]
x.shape,y.shape
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10, random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
x_train=x_train.values.reshape(37800,28,28,1)

x_test=x_test.values.reshape(4200,28,28,1)

#x_train.shape
x_train=x_train/255

x_test=x_test/255
#from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from keras.utils import to_categorical 

from keras.constraints import max_norm
classes=10

y_train=to_categorical(y_train,num_classes = classes)

y_test=to_categorical(y_test,num_classes = classes)
y_test.shape, y_train.shape,x_train.shape,x_test.shape
model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1),padding='Same'))

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',padding='Same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='Same'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='Same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

#model.add(Dense(512, activation='relu',kernel_constraint=max_norm(3)))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=30,verbose=1,validation_data=(x_test,y_test))
x_test_data=x_test_data.values.reshape(28000,28,28,1)
results = model.predict(x_test_data,batch_size=128,verbose=1)
results.shape
results
results = np.argmax(results,axis = 1)



#results = pd.Series(results,name="Label")
results
results.shape
results = pd.Series(results,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("Avinash.csv", index=False, header=True)
