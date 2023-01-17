# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import numpy as np

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Convolution2D,Dropout,Dense,Flatten,MaxPooling2D,Conv2D,Conv1D

import pandas as pd





# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y_1_train=train.pop('label')
train=train/255

test=test/255

train=train.values

test=test.values

test=test.reshape((28000,28,28,1))

train=train.reshape((42000,28,28,1))
l2=keras.regularizers.l2()

model=Sequential([Conv2D(6,5,strides=1,padding='valid',activation='relu',input_shape=(28,28,1))

                 ,MaxPooling2D((2,2),strides=2,padding='valid')

                 ,Conv2D(16,5,strides=1,padding='valid',activation='relu')

                 ,MaxPooling2D((2,2),strides=2,padding='valid') 

                 ,Flatten() 

                 ,Dense(120,activation='relu')

                 ,Dense(84,activation='relu')

                ,Dense(10,activation='softmax')

                 ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train,y_1_train,batch_size=500,verbose=1,epochs=50)
y_pred=model.predict_classes(test)
ID=list(range(1,28001))

len(ID),len(y_pred)

ID=np.array(ID)

ID=ID.reshape(-1,1)
cols = ['ImageId', 'Label']

submit_df = pd.DataFrame(np.hstack((ID,y_pred.reshape(-1,1))), columns=cols)
submit_df.to_csv('submission.csv', index=False)
submit_df