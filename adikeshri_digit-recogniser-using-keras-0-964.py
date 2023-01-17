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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print(train.info())

print(train.head())
print(train.isnull().any().describe())

print(train.shape)
d=pd.get_dummies(train['label'])

names=['0','1','2','3','4','5','6','7','8','9']

c=0

for col in names:

    train[col]=d[c]

    c+=1

train.drop('label',axis=1,inplace=True)

print(train.info())
X=train.drop(names,axis=1)

Y=train[names]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
from keras.models import Sequential

from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,GlobalAveragePooling2D

from keras.optimizers import SGD

model=Sequential()

model.add(Dense(64, activation='relu',input_shape=(784,)))

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist=model.fit(x_train,y_train,epochs=10,batch_size=32)
model.evaluate(x_test,y_test)
test.head()
ans=model.predict(test)

ans=pd.DataFrame(ans)

ans=ans.idxmax(axis=1)

a=[]

id=[]

for i in range(0,28000):

    a.append(ans[:][i])

    id.append(i+1)
sub=pd.DataFrame({'ImageId':id,'Label':a})

sub.to_csv('output.csv',index=False)