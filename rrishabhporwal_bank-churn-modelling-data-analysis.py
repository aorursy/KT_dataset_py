
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
file = '../input/Churn_Modelling.csv'

#importing the dataset
dataset = pd.read_csv(file)

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values
# Encoding the categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]
# Dataset into training and test set 
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing keras libraries and packages
import keras

from keras.models import Sequential
from keras.layers import Dense
# Initializing the ANN
classifier = Sequential()
# Adding the input layer and he first hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform",input_dim=11))
# Addding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
# Compiling the ANN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the training dataset
classifier.fit(X_train, y_train,batch_size=10, epochs=100)
# Predicting the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm