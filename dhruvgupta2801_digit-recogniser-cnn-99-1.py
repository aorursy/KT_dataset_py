# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/train.csv')

train_data.head()
train_data.shape
train_data.info()
train_data.describe()
test_data=pd.read_csv('../input/test.csv')

test_data.head()
test_data.info()
plt.imshow(train_data.iloc[0,1:].values.reshape(28,28),cmap='gray')

#reshape(28,28),cmap='gray')
plt.imshow(train_data.iloc[3,1:].values.reshape(28,28),cmap='gray')
train_data['label'].value_counts().plot(kind='bar')
from keras.layers import Dense,Activation,Conv2D,BatchNormalization,Flatten,MaxPool2D,Dropout

from keras.models import Sequential

from keras.utils import np_utils

from keras.regularizers import l1,l2,l1_l2
from sklearn.model_selection import train_test_split

X=train_data.drop('label',axis=1)

y=train_data['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=40)
Y=np_utils.to_categorical(y)

Y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=40)
print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
X_train=np.asarray(X_train)

X_train=X_train.reshape(-1,28,28,1)

X_test=np.asarray(X_test)

X_test=X_test.reshape(-1,28,28,1)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
model=Sequential()

model.add(Conv2D(512,(3,3),input_shape=(28,28,1)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,3))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(16,3))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.3))

#model.add(Dense(128))

#model.add(Activation('relu'))

#model.add(Dense(50))

#model.add(Activation('relu'))

#model.add(Dropout(0.2))

#model.add(Dense(40))

#model.add(Activation('relu'))

#model.add(Dense(40))

#model.add(Activation('relu'))

#model.add(Dense(20))

#model.add(Activation('relu'))

#model.add(Dense(20))

#model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax'))

model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
z=model.fit(x=X_train,y=y_train,batch_size=1000,epochs=20,verbose=2,validation_data=(X_test,y_test))
plt.figure(figsize=(5,5))

plt.plot(z.history['acc'],color='b',label='Training Set')

plt.plot(z.history['val_acc'],color='r',label='Test Set')

plt.legend()

plt.ylabel('Accuracy')

plt.show()
plt.figure(figsize=(5,5))

plt.plot(z.history['loss'],color='b',label='Training Set')

plt.plot(z.history['val_loss'],color='r',label='Test Set')

plt.ylabel('Loss')

plt.legend()

plt.show()
test_data=np.asarray(test_data).reshape(-1,28,28,1)

test_data.shape

pred=model.predict(test_data)

pred=np.round(pred,0)

predictions=[]

for i in range(pred.shape[0]):

    count=0

    for j in range(pred.shape[1]):

        if pred[i,j]==1:

            predictions.append(j)

            count=1

    if count==0:

        predictions.append(0)
predictions=np.asarray(predictions)

predictions.shape
submission=pd.DataFrame(predictions,columns=['Label'])

submission['ImageId']=np.arange(1,test_data.shape[0]+1)

submission=submission[['ImageId','Label']]

submission.head()



#submission.to_csv('Submission.csv')
submission.set_index('ImageId',inplace=True)
submission.head()

submission.to_csv('Submission.csv')