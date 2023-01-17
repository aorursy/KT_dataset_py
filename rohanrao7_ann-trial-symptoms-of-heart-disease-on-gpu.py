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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time
#importing dataset 

dataset = pd.read_csv('../input/heart.csv')

x=dataset.iloc[:,1:13].values

y=dataset.iloc[:,13].values

start = time.time()
#splitting the dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
#feature scaling is a must in nueral network, as we have large number of data leading to heavy computational

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#lets create an ANN

#import the keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

#Using TensorFlow backend.

#initialising ANN

classifier = Sequential()

#Adding the input layer and first hidden layer

classifier.add(Dense(activation="relu", input_dim=12, units=6, kernel_initializer="uniform"))

#adding the second hidden layer

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#Adding the output layer

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train, batch_size=10, nb_epoch = 100)
#prediction

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)
#confusionmatrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
TP = cm[1,1]

FP = cm[0,1]

TN = cm[1,0]

FN = cm[0,0]

Total = (TP+FP+TN+FN)

Acc = (TP+FN)/Total

Acc
end = time.time()

print(end - start)