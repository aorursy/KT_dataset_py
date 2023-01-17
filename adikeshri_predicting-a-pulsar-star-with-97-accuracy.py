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
data=pd.read_csv('../input/pulsar_stars.csv')

data.sample(5)

cols=data.columns

for col in cols:

    print('Column:' + str(col))

    print(np.corrcoef(data['target_class'],data[col]))
d=pd.get_dummies(data['target_class'])

data=pd.concat([data,d],axis=1)

data.drop('target_class',axis=1,inplace=True)

data.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data.drop([0,1],axis=1),data[[0,1]],test_size=0.2)
!pip install keras-metrics
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import adam

import keras_metrics as km

model=Sequential()

for i in range(0,4):

    model.add(Dense(100,input_shape=(8,),activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64,input_shape=(8,),activation='relu'))

model.add(Dense(64,input_shape=(8,),activation='relu'))

model.add(Dense(64,input_shape=(8,),activation='relu'))

model.add(Dense(64,input_shape=(8,),activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(32,input_shape=(8,),activation='relu'))

model.add(Dense(32,input_shape=(8,),activation='relu'))

model.add(Dense(32,input_shape=(8,),activation='relu'))

model.add(Dense(32,input_shape=(8,),activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))

print('Model summary: ')

model.summary()

print('')

opt = adam(lr=0.001, decay=1e-6)

model.compile(

    loss='categorical_crossentropy',

    optimizer=opt,

    metrics=['accuracy',km.precision(),km.recall()])

print('Model fitting...')

model.fit(x_train,y_train, epochs=20, batch_size=32)

score=model.evaluate(x_test,y_test)

print('Accuracy(on Test-data): ' + str(score[1]))
