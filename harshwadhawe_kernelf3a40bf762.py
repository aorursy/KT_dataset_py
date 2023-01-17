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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout

from tensorflow.keras.models import Model
y_train = train['label']

y_train = y_train.to_numpy()

y_train.shape
K = len(set(y_train))

print(K)
train.drop(columns=['label'],inplace=True)

x_train = train.to_numpy()

x_train.shape
#test.drop(columns=['id'],inplace=True)

x_test = test.to_numpy()

x_train,x_test = x_train/255.0,x_test/255.0

x_test.shape
x_train = x_train.reshape(42000,28,28)

x_test = x_test.reshape(28000,28,28)



x_train = np.expand_dims(x_train, -1)

x_test = np.expand_dims(x_test, -1)

print(x_test.shape)

print(x_train.shape)
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout

from tensorflow.keras.models import Model

# Build the model using the functional API

i = Input(shape=x_train[0].shape)

x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)

x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Flatten()(x)

x = Dropout(0.2)(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(K, activation='softmax')(x)



model = Model(i, x)
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)
r = model.fit(x_train,y_train,epochs=16)
import matplotlib.pyplot as plt

plt.plot(r.history['loss'],label='Loss')

plt.legend()
pred = model.predict(x_test)

y = pd.DataFrame(pred)

y.head()
y = pd.DataFrame(pred)

y = pd.DataFrame(y.idxmax(axis = 1))

y.index.name = 'ImageId'

y = y.rename(columns = {0: 'Label'}).reset_index()

y['ImageId'] = y['ImageId'] + 1
y.to_csv('mnist_kaggle.csv',index=False)