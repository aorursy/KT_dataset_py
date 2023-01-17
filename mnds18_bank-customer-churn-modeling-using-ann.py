# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#https://www.kaggle.com/barelydedicated/ann-bank-customer-classification



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import theano

import tensorflow

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing the dataset

dataset = pd.read_csv('../input/Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values # Credit Score through Estimated Salary

y = dataset.iloc[:, 13].values # Exited

print(dataset.shape)

print(X.shape)

print(y.shape)
dataset.head()
X[:]
# Encoding categorical (string based) data. Country: there are 3 options: France, Spain and Germany

# This will convert those strings into scalar values for analysis

print(X[:8,1], '... will now become: ')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_X_country_encoder = LabelEncoder()

X[:,1] = label_X_country_encoder.fit_transform(X[:,1])

print(X[:8,1])
# We will do the same thing for gender. this will be binary in this dataset

print(X[:6,2], '... will now become: ')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_X_gender_encoder = LabelEncoder()

X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])

print(X[:6,2])
print(X[:])

print(X.shape)
# Converting the string features into their own dimensions. Gender doesn't matter here because its binary

countryhotencoder = OneHotEncoder(categorical_features = [1]) # 1 is the country column

X = countryhotencoder.fit_transform(X).toarray()
print(X.shape)
X = X[:,1:]
print(X.shape)
# Splitting the dataset into the Training and Testing set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train.shape

# print(len(y_train))
X_train
import keras # Test out Theano when time permits as well
from keras.models import Sequential

from keras.layers import Dense
# Initializing the ANN

classifier = Sequential()
# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)

classifier.add(Dense(activation = 'relu', input_dim = 11, units=6, kernel_initializer='uniform'))
# Adding the second hidden layer

# Notice that we do not need to specify input dim. 

classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 
# Adding the output layer

# Notice that we do not need to specify input dim. 

# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)

# We use the sigmoid because we want probability outcomes

classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform')) 
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
print(len(X_train))

print(len(y_train))
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
y_pred = classifier.predict(X_test)

print(y_pred)
y_pred = (y_pred > 0.5)

print(y_pred)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')