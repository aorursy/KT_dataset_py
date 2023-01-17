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
import tensorflow as tf

from tensorflow import keras



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras import backend as K
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

y_train = train['label']

X_train = train.drop(labels = ['label'],axis=1)

print(X_train.shape)
#Show image of training data



plt.figure(figsize = (10,10))

rand_indexes = np.random.randint(0, X_train.shape[0], 16)

  

for index,im_index in enumerate(rand_indexes):

  plt.subplot(4, 4, index+1)

  plt.imshow(X_train.values[im_index].reshape(28, 28), cmap='Blues', interpolation ='none')

  plt.title('Class %d' %y_train[im_index])

 

plt.tight_layout()
#Prepare dataset



X_train = X_train.astype('float32') / 255

test = test.astype('float32')/255





X_train = X_train.values.reshape(-1,28,28,1)



test = test.values.reshape(-1,28,28,1)



y_train = keras.utils.to_categorical(y_train)

num_classes = y_train.shape[1]



from sklearn.model_selection import train_test_split

import time



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=int(time.time()))







#Build model



model = Sequential()



#1st convulation layer



model.add(Conv2D(32, kernel_size= (3,3), padding = 'same', activation ='relu', input_shape=(28,28,1)))



model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Dropout(0.25))





#2nd convulation layer



model.add(Conv2D(64, kernel_size =(3, 3), padding='same', activation ='relu'))



model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Dropout(0.25))



#Fully connected layer



model.add(Flatten())



model.add(Dense(128, activation='relu'))



model.add(Dropout(0.2))



model.add(Dense(num_classes, activation='softmax'))
model.summary()
#compile the model

model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])
#fit the model





num_epochs = 50



model.fit(X_train, y_train, batch_size = 512,  epochs= num_epochs, verbose =1)
result = model.evaluate(X_test, y_test, verbose = 0)

print('Accuracy: ', result[1])