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
import numpy as np 

import keras 

import pandas as pd

from keras.datasets import fashion_mnist

from keras.layers import Dense, Flatten

from keras.models import Sequential

from keras import metrics 

from keras import optimizers
df = pd.read_csv('../input/train.csv')
y_train = df['label'].values.reshape(df.shape[0], 1)

X_train = df.drop('label', axis = 1).values

print(X_train.shape)
X_train = X_train / 255.

num_classes = 10 
y_train = keras.utils.to_categorical(y_train, num_classes)
model = Sequential()

model.add(Dense(128, input_shape = (784,), activation = 'linear'))

model.add(Dense(256, activation = 'relu'))

model.add(Dense(512, activation = 'relu'))

model.add(Dense(1024,activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))
sgd = optimizers.SGD(lr = 0.12)

model.compile(loss = keras.losses.categorical_crossentropy,

              optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 256, epochs = 50)
df2 = pd.read_csv('../input/test.csv')

X_test = df2.values

predictions = model.predict_classes(X_test)
submission = pd.DataFrame({'ImageId':range(1,28001), 'Label':predictions})

submission.to_csv('submission.csv',index=False)

print('Save to file submission.csv')