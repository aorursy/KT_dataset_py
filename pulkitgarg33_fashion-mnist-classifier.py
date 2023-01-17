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
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
#importing the dataset
dataset = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')
x_test = np.array(test.iloc[: , 1:])
y_test = keras.utils.to_categorical(np.array(test.iloc[: , 0]))


x = np.array(dataset.iloc[: , 1:])
y = keras.utils.to_categorical(np.array(dataset.iloc[: , 0]))
image_rows = 28
image_colm = 28
num_classes = 10

from sklearn.model_selection import train_test_split
x_train , x_val , y_train , y_val = train_test_split( x , y , test_size = 0.2)
x_train = x_train.reshape( x_train.shape[0] , image_rows , image_colm , 1)
x_test = x_test.reshape( x_test.shape[0] , image_rows , image_colm , 1)
x_val = x_val.reshape( x_val.shape[0] , image_rows , image_colm , 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train = x_train / 255
x_test = x_test / 255
x_val = x_val / 255

# now making the CNN

#initialising the CNN
from keras.models import Sequential
classifier = Sequential()

#Adding the first convolution layer
from keras.layers import Conv2D
classifier.add( Conv2D ( 32, (3,3) , activation = 'relu' , input_shape = (image_rows , image_colm , 1)))
#adding the pooling layer
from keras.layers import MaxPooling2D
classifier.add( MaxPooling2D ( pool_size = (2,2) ))
#Flatteing
from keras.layers import Flatten
classifier.add( Flatten())

#now our CNN layer is ready , now feedinf it to an ANN
from keras.layers import Dense
classifier.add(Dense( output_dim = 64 , activation = 'relu'))
classifier.add(Dense( output_dim = 10 , activation = 'softmax'))

#now compiling our CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#now fitting our model
history = classifier.fit(x_train, y_train, batch_size=200, epochs= 15, verbose=1, validation_data=(x_val, y_val))

# analyzing the performance of our model
score = classifier.evaluate( x_test , y_test )
#making predictions
y_pred = classifier.predict( x_test)

# storing the results
y_pred_1 = np.argmax( y_pred , axis = 1)
y_test_1 = np.argmax( y_test , axis = 1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_test_1 , y_pred_1)
cm