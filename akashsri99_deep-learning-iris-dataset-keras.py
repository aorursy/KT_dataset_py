# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
##!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Thu Jul 20 23:48:47 2017



@author: akashsrivastava

"""



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from keras.utils import to_categorical





#os.chdir('/Users/akashsrivastava/Desktop/MachineLearning/kaggle/iris-keras')



dataset = pd.read_csv('../input/Iris.csv')



dataset
#Plotting the pairwise relationship of different parameters



import seaborn as sns

sns.set(style="ticks")

sns.set_palette("husl")

sns.pairplot(dataset.iloc[:,1:6],hue="Species")
#Splitting the data into training and test test

X = dataset.iloc[:,1:5].values

y = dataset.iloc[:,5].values



from sklearn.preprocessing import LabelEncoder

encoder =  LabelEncoder()

y1 = encoder.fit_transform(y)



Y = pd.get_dummies(y1).values





from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0) 







#Defining the model 



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD,Adam





model = Sequential()



model.add(Dense(10,input_shape=(4,),activation='tanh'))

model.add(Dense(8,activation='tanh'))

model.add(Dense(6,activation='tanh'))

model.add(Dense(3,activation='softmax'))



model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])



model.summary()
#fitting the model and predicting 

model.fit(X_train,y_train,epochs=100)

y_pred = model.predict(X_test)



y_test_class = np.argmax(y_test,axis=1)

y_pred_class = np.argmax(y_pred,axis=1)









#Accuracy of the predicted values

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test_class,y_pred_class))

print(confusion_matrix(y_test_class,y_pred_class))


