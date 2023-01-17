# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def prep_train_data(raw):

    out_y = keras.utils.to_categorical(raw.label,num_classes=10)

    number_of_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_tensor = x_as_array.reshape(number_of_images,28,28,1)

    out_x = (x_tensor-128)/255

    return out_x, out_y



def prep_test_data(raw):

    number_of_images = raw.shape[0]

    x_as_array = raw.values

    x_tensor = x_as_array.reshape(number_of_images,28,28,1)

    out_x = (x_tensor-128)/255

    return out_x

    
raw_train_data = pd.read_csv("../input/train.csv")

raw_test_data = pd.read_csv("../input/test.csv")
raw_test_data.shape[0]
x_train, y_train = prep_train_data(raw_train_data)

x_test= prep_test_data(raw_test_data)
model = Sequential()

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28, 28, 1)))

model.add(Conv2D(20,kernel_size=(4,4),activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax')) #10 is the number of classes

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.fit(x_train, y_train,

          batch_size=128,

          epochs=2,

          validation_split = 0.2)
output = pd.DataFrame({'Label':model.predict_classes(x_test)})

output.index += 1

output.to_csv('out.csv',index_label='ImageId')