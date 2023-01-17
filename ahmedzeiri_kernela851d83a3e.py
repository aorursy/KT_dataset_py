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
import pandas as pd

from sklearn.model_selection import train_test_split



train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

#est = pd.read_csv(r'C:\Users\HP\Desktop\test.csv')



#y_train = train.values[:,0]

#y_test = train.values[0:5000,0]

#x_test = test.values[:,1:]

#x_train = train.values[: , 1:]



x = train.values[:, 1:]

y = train.values[:,0]

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)





x_train = x_train.reshape(48000 , 28 , 28)

x_test = x_test.reshape(12000, 28 , 28)

y_test = y_test

y_train = y_train



from keras.datasets import mnist

from keras.utils import np_utils

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K





batch_size = 128

epochs = 5



# Lets store the number of rows and columns



img_rows = x_train[0].shape[0]

img_cols = x_train[1].shape[0]



# Getting our date in the right 'shape' needed for Keras

# We need to add a 4th dimenion to our date thereby changing our

# Our original image shape of (60000,28,28) to (60000,28,28,1)



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)







# store the shape of a single image 

input_shape = (img_rows, img_cols, 1)



# change our image type to float32 data type

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



# Normalize our data by changing the range from (0 to 255) to (0 to 1)

x_train /= 255

x_test /= 255



print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')





# Now we one hot encode outputs

y_train = np_utils.to_categorical(y_train )

y_test = np_utils.to_categorical(y_test)



# Let's count the number columns in our hot encoded matrix 

#print ("Number of Classes: " + str(y_test.shape[1]))



num_classes =10

num_pixels = x_train.shape[1] * x_train.shape[2]





# create model

model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss = 'categorical_crossentropy',

              optimizer = keras.optimizers.Adadelta(),

              metrics = ['accuracy'])



print(model.summary())



history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
