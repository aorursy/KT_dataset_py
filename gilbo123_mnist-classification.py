# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stat

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils





np.random.seed(123)  # for reproducibility



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#print(train.shape)

#separate labels, then reshape data to 2D

train_labels = train["label"]

train_data = train.loc[:, train.columns != 'label']

train_data = train_data.values.reshape((42000, 28, 28, 1)).astype('float32')

#test data has no labels so just reshape

test_data = test.values.reshape((28000, 28, 28, 1)).astype('float32')



#change from 0-255 => 0.0-1.0

train_data /=  255

test_data /=  255



#convert the labels to 'one-hot' format

train_labels = np_utils.to_categorical(train_labels, 10)





#set up model

model = Sequential()

#input layers

model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.10))

#output layers

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

#compile

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=32, epochs=10, verbose=1)

#predict

predictions = model.predict_classes(test_data, verbose=0)

submission=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submission.to_csv("predictions.csv", index=False, header=True)
