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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(3)
from sklearn.model_selection import train_test_split

y = train['label']
x = train.drop('label',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=13)
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import keras
from keras import backend as K
K.set_image_dim_ordering('th')


x_train = x_train.as_matrix()
x_test = x_test.as_matrix()

sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

x_train = x_train.reshape(x_train.shape[0],1,28,28).astype('float32')
x_test = x_test.reshape(x_test.shape[0],1,28,28).astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
x_train.shape[0]
model = Sequential()

model.add( Conv2D(filters=32, kernel_size=(10,10), input_shape=(1,28,28), activation='relu' ) )
model.add( MaxPooling2D(pool_size=(2,2)) )
model.add( Dropout(0.2) )
model.add( Flatten() )
model.add( Dense(128, activation='relu') )
model.add( Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128, verbose=1)

_test = sc.fit_transform(test)

_test = _test.reshape(_test.shape[0],1,28,28).astype('float32')
_test_result= model.predict_classes(_test)
_test_result
len(_test_result)
submission = pd.DataFrame(index=test.index, columns=['ImageId','Label'])

for i in range(len(_test_result)):
    submission['Label'][i] = _test_result[i]
submission['ImageId'] = test.index   
submission.head(3)

from matplotlib.pyplot import imshow
%matplotlib inline
__test = test
_test2 = _test2.reshape(_test2.shape[0],28,28).astype('float32')

imshow(_test2[0])




