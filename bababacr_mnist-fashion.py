# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the required Keras modules containing model and layers

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train=pd.read_csv("../input/fashion-mnist_train.csv")

data_test=pd.read_csv("../input/fashion-mnist_test.csv")

#data_train.head(10)

#data_test.head()
import matplotlib.pyplot as plt

for i in range(3):

    reshaped=np.reshape(data_test.iloc[i,1:].values,(28,28))

    reshaped=reshaped/255

    plt.figure()

    plt.title("label {}".format(data_test.iloc[i,0]))

    plt.imshow(reshaped)

    

#converting data into np array (for keras)

y_train = np.array(data_train.iloc[:,0])

X_train = np.array(data_train.iloc[:,1:])



#reshaping data

x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

x_train = x_train/255

#defining the model

model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))



#compiling

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])



#fitting

model.fit(x=x_train,y=y_train, epochs=15)
y_test = np.array(data_test.iloc[:,0])

X_test = np.array(data_test.iloc[:,1:])



x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

x_test =x_test/255



#evalutation

model.evaluate(x_test, y_test)

# we achieve 92%, we can do better