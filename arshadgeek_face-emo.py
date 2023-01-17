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

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical



import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt
data = pd.read_csv('../input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')
data.head(5)
data.shape
print(data['emotion'].unique())

data['Usage'].unique()
train = data.where(data['Usage']== 'Traning')
train.shape
train_split = data['Usage'] == 'Training'

test_split  = data['Usage'] == 'PublicTest'

validation_split = data['Usage'] == 'PrivateTest'



train = data[train_split]

test  = data[test_split]

validation = data[validation_split]
train.head(5)

y_train = train['emotion'].values

y_test  = test['emotion'].values



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
X_train = train['pixels'].str.split(' ', expand=True).rename(columns = lambda x: "pixel"+str(x+1))

X_test = test['pixels'].str.split(' ', expand=True).rename(columns = lambda x: "pixel"+str(x+1))
X_train
X_train = X_train.values.reshape((28709, 48, 48, 1))

X_test = X_test.values.reshape((3589 , 48 , 48 , 1))
model = Sequential()

 

#1st convolution layer

model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))

model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

 

#2nd convolution layer

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

 

#3rd convolution layer

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

 

model.add(Flatten())

 

#fully connected neural networks

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

 

model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=64)
accuracy = model.evaluate(x=X_test,y=y_test,batch_size=64)

print("Accuracy: ",accuracy[1])
plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['accuracy', 'loss'], loc='upper left')