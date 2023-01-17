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
df_train=pd.read_csv('../input/train.csv')
df_train.head()
df_test=pd.read_csv('../input/test.csv')
df_train.shape
X_train=df_train.iloc[:,1:].values
X_train
X_train=X_train.astype(float)
X_train /=255
y_train=df_train.iloc[:,0].values
y_train
X_test=df_test.values
X_test
X_test=X_test.astype(float)
X_test /=255
import keras
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import Adam,SGD
sgd=SGD(momentum=0.9,decay=1e-6)
from keras.losses import categorical_crossentropy as ccentropy
from keras.utils import to_categorical
y_train=to_categorical(y_train,10)
y_train
model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss=ccentropy,optimizer=sgd,metrics=['accuracy'])
model.fit(x=X_train,y=y_train,batch_size=128,epochs=20)
model.predict_classes(X_test)
s=pd.DataFrame(data=model.predict_classes(X_test),columns=['Label'])
s.head()
s.index=s.index+1
s.head()
my_submission = pd.DataFrame({'ImageId': s.index, 'Label': s.Label})
my_submission.head()
my_submission.to_csv('submission.csv', index=False)
