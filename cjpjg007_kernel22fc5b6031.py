# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

data=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data.append(pd.read_csv(os.path.join(dirname, filename)))

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=data[0]

test=data[1]

sub=data[2]

train.head()
x=np.array(train.drop('label',axis=1))

y=train['label']

print(x.shape,test.shape)

test.head()
x=x.reshape((42000,28,28,1))
from keras.utils import to_categorical

y=to_categorical(y)
from keras.layers import Conv2D,Flatten,Dense,Dropout

from keras.models import Sequential
model=Sequential()

model.add(Conv2D(20, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(Dropout(0.5))

model.add(Conv2D(20,kernel_size=3,activation='relu'))

model.add(Dropout(0.5))

model.add(Conv2D(18,kernel_size=2,activation='relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128))

model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,batch_size=128,epochs=8,validation_split=0.2)
predictions=np.argmax(model.predict(np.array(test).reshape(28000,28,28,1)),axis=1)

predictions
sub.shape,sub.columns
sub['Label']=predictions
sub.to_csv("/kaggle/output",index=False)