# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Reshape

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD

from keras.utils.np_utils import to_categorical

import keras.optimizers as op



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#from numpy._distributor_init import NUMPY_MKL  # requires numpy+mkl

from sklearn import datasets, linear_model

import seaborn as sns

from sklearn.model_selection import train_test_split



batch_size = 1000

num_classes = 10

epochs = 50



train = pd.read_csv("../input/train.csv")



train.head()

train_X = train.iloc[:,1:]

train_Y = train.iloc[:,0]



train_X.head()

train_Y.head()



x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.33, random_state=42)



#Normalize from 0 - 255 to 0-1

x_train = x_train / 255

x_test = x_test / 255

x_train.describe()



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

y_train = to_categorical(y_train, num_classes)

y_test = to_categorical(y_test, num_classes)



x_train = x_train.values

x_test = x_test.values



# reshape to be [samples][pixels][width][height]

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')



x_train.shape



del(model)

model = Sequential()

# input: 28x28 images with 1 channels -> (28, 28, 1) tensors.

# this applies 32 convolution filters of size 3x3 each.



model.add(Conv2D(filters = 32, 

                 kernel_size = (3, 3), 

                 activation='relu', 

                 data_format='channels_first', 

                 input_shape=(1, 28, 28)))

                 

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])



model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test, batch_size=batch_size)



test = pd.read_csv("../input/test.csv")

test = test/255

test = test.values

test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')

prediction = model.predict(test, batch_size=batch_size)



prediction_df = pd.DataFrame(prediction)

prediction_label_df = prediction_df.idxmax(axis = 1)

prediction_label_df.to_csv("prediction_label_df.csv",index = True)