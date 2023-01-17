# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.misc import imread # reading the images

from sklearn.metrics import accuracy_score # predicting accuracy

import keras # NN API

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

seed=100

r=np.random.RandomState(seed) # for introducing reandomness in the training data

# input of the data

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

x_train=(train.ix[:,1:].values).astype('float32')#removing the label

y_train=(train.ix[:,0].values).astype('float32')#extracting the label

x_test=(test.ix[:,:].values).astype('float32')

x_train=x_train.reshape(x_train.shape[0],28,28,1)#reshaping to feed as input to NN

x_test=x_test.reshape(x_test.shape[0],28,28,1)

x_train=x_train/255# pixel values range from 0-255, to normalize the values

x_test=x_test/255

y_train= keras.utils.np_utils.to_categorical(train.label.values)# to convert to categories

#splitting into training and validation data

split_size=int(x_train.shape[0]*0.7)

t_x,val_x=x_train[:split_size],x_train[split_size:]

t_y,val_y=y_train[:split_size],y_train[split_size:]



#importing libraries for NN

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

#defining hyperparameters

epochs=10

batch_size=128

#creating model and adding layers

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=( 28, 28,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))

#compiling the model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#training model using training data

trained_model=model.fit(t_x,t_y,nb_epoch=epochs,batch_size=batch_size,validation_data=(val_x,val_y))

#predicting for the test datasets

pred=model.predict_classes(x_test)

#printing predicted values

print(pred)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.