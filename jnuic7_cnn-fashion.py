# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import MaxPooling2D,Dropout,Flatten

from keras.layers.convolutional import Conv2D

from keras.utils import np_utils

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
datatrain = pd.read_csv("../input/fashion-mnist_train.csv")

datatest = pd.read_csv("../input/fashion-mnist_test.csv")

X_train,X_test = datatrain.iloc[:,1:].values,datatest.iloc[:,1:].values

y_train,y_test = datatrain.label.values,datatest.label.values

X_train,X_test = X_train/255,X_test/255

X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

cat_size = y_test.shape[1]
model = Sequential()

model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(cat_size,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=2)
score = model.evaluate(X_test,y_test)

score[1]
#Lets look at an example for fun

plt.imshow(X_test[0].reshape(28,28),cmap='gray')
#looks like a shirt, lets test the model

#The model could be improved by augmenting the photos to capture different orientations.

#Also the network topology could be made deeper, more to come

print(np.round(model.predict(X_test[0].reshape(1,28,28,1))))

print(y_test[0])