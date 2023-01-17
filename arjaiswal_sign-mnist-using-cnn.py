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
import matplotlib.pyplot as plt
sign_train = pd.read_csv("../input/sign_mnist_train.csv")
sign_test = pd.read_csv("../input/sign_mnist_test.csv")
#Getting the training and testing data and output
x_test = sign_test.iloc[:, 1:].values
y_test = sign_test.iloc[:, 0].values
x_train = sign_train.iloc[:, 1:].values
y_train = sign_train.iloc[:, 0].values
from PIL import Image
#To see the image
img = Image.fromarray(x_train[0, :].reshape(28, 28).astype('uint8'))
img.show()
#OnceHotEncoding for y_train
from sklearn.preprocessing import OneHotEncoder
y_train= y_train.reshape(-1,1)
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()

y_test= y_test.reshape(-1,1)
ohe = OneHotEncoder()
y_test = ohe.fit_transform(y_test).toarray()
#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Reshaping the data to fit the original size of the image
x_train = np.reshape(x_train, (27455, 28, 28, 1))
x_test = np.reshape(x_test, (7172, 28, 28, 1))
#Training the model now
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier= Sequential()
#Add Convolutional layer
classifier.add(Convolution2D(16, 3,3, input_shape=(x_train.shape[1:]), activation='relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
#Flattening
classifier.add(Flatten())
#Adding Hiden Layers
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=24, activation="sigmoid"))
#Compiling the model
classifier.compile(optimizer="adam", loss="categorical_crossentropy", 
                   metrics=["accuracy"])
#Now training the model
classifier.fit(x_train, y_train, batch_size = 10, epochs = 2)
#Predicting the data with test input
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
#Finding the accuracy of model
count = 0
for i in range(0, y_pred.shape[0]):
    for j in range(0, y_pred.shape[1]):
        if y_pred[i][j] == y_test[i][j]:
            count = count+1
        else:
            continue    
#print(count) 
print("Accurancy: "+str(count/(7172*24)))
