# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import math

import cv2
train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

import tensorflow 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Convolution2D, MaxPooling2D

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np 
train.head()
f=train.iloc[30,1:].values.reshape(28,28)
import matplotlib.pyplot as plt
plt.imshow(f)
df_x = train.iloc[:,1:].values.reshape(len(train),28,28,1).astype('float32')



#Storing the labels in y

y = train.iloc[:,0].values

y
import tensorflow.keras.utils as tfk
df_y = tfk.to_categorical(y, num_classes=10, dtype='int64')
df_x = np.array(df_x)

df_y = np.array(df_y)
df_x
df_y
y
df_x.shape
df_y.shape
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#CNN model

model = Sequential()

model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100))

model.add(Dropout(0.3))

model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=8,verbose=2)
model.predict(x_test)
model.predict_classes(x_test)
train.iloc[100:105:,1:]
train.iloc[100:110:,:1]
var =train.iloc[50:60:,1:].values.reshape(10,28,28,1).astype('float32')
model.predict_classes(var)
plt.imshow(x_train[0][:,:,0])
plt.imshow(x_test[0][:,:,0])