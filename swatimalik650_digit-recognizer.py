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
import os

print(os.listdir("../input"))
import numpy as np # linear algebra

import pandas as pd 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train

x_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values


# Making sure that the values are float so that we can get decimal points after division

x_train = x_train.astype('float32')

test = test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.

x_train /= 255

test /= 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)



input_shape = (28, 28, 1)
test = test.values.reshape(test.shape[0], 28, 28, 1)
test.shape
from keras.models import Sequential
from keras.layers import Convolution2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D

from keras.layers import Dense
classifier = Sequential()
classifier.add(Convolution2D(32,3,3,input_shape = (28,28,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128,activation='relu'))
classifier.add(Dense(output_dim = 10,activation='softmax'))
classifier.compile(optimizer = 'adam',loss ='sparse_categorical_crossentropy',metrics = ['accuracy'])
classifier.fit(x=x_train,y=(y_train), epochs=50,batch_size = 32)

predictions = classifier.predict_classes(test, verbose=1)

submission = pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions})
submission.to_csv('mnist_submission', index = False)