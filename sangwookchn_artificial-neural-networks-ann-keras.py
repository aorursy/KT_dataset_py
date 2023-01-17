import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/Churn_Modelling.csv')

#include relevant columns within x and y
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
dataset.head()
x.head()
#deal with categorical data --> encode them

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 1] = labelencoder_x.fit_transform(x.iloc[:, 1]) #applying on Geography
x.head()
#apply encoder on Gender as well
labelencoder_x_2 = LabelEncoder()
x.iloc[:, 2] = labelencoder_x_2.fit_transform(x.iloc[:, 2]) #applying on Gender
x.head()
#One hot encoding. 

from keras.utils import to_categorical
encoded = pd.DataFrame(to_categorical(x.iloc[:, 1]))
#no need to encode Gender, as there are only two categories

x = pd.concat([encoded, x], axis = 1)
x.head()
#Dropping the existing "geography" category, and one of the onehotcoded columns.

x = x.drop(['Geography', 0], axis = 1)
x.head()
#train and test set split, and feature scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense #to add layers

#there is no rule on how many nodes each hidden layer should have
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#init --> initialize weights according to uniform distribution
#input_dim is required for the first hidden layer, as it is the first starting point. --> number of nodes.
#output_dim --> number of nodes of the hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#input_dim --> remove it as it already knows what to expect.

#the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#output_dim should be 1, as output is binary outcome, and activation should be 'sigmoid'
#If dependent variables have more than two categories, use activation = 'softmax'

#compile the model --> backpropagation -> gradient descent
classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])
#optimizer = algorithm to find the optimal set of weights in ANN
#loss = functions that should be optimized. if more than two categories, use "categorical_crossentropy"
#metrics = criterion used to calculate the performance of the model.
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 20)
#batch_size = the number of observations after which you want to update the weights
#           batch size and epochs should be tuned through experiments.
#epoch = going through the whole dataset
#predicting the results

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #to classify each probability into True or False

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print (cm, '\n\n', y_pred[:5, :])
#accuracy
print ((1548 + 139)/2000)