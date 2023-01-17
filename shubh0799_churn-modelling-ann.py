#Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Importing the data set

dataset = pd.read_csv('/kaggle/input/Churn_Modelling.csv')

dataset
# Taking dataset in numpy

dset = dataset.iloc[:, 3:].values
# Encoding categorical Variables

from sklearn.preprocessing import LabelEncoder

le_country = LabelEncoder()

dset[:, 1] = le_country.fit_transform(dset[:, 1])

le_country = LabelEncoder()

dset[:, 2] = le_country.fit_transform(dset[:, 2])

np.set_printoptions(threshold=np.inf)
dset
# Creating dummy variables

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories = 'auto'), [1, 2])], remainder='passthrough')

dset = ct.fit_transform(dset)

np.set_printoptions(threshold=np.inf)

print(dset)
#Splitting into x and y

x = dset[:, 1:13]

y = dset[:, 13:14]

print(x)

print(y)
x = np.delete(x,[3],1)

print(x)

# First two coulmn are of country and then the third one is of gender
# mising data sklearn not found skipping that and Categorical Variables

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

"""print('x train= ',x_train)

print('x test= ',x_test)

print('y train= ',y_train)

print('y test= ',y_test)"""
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
#Importing the keras libraries

import keras

from keras.models import Sequential

from keras.layers import Dense
# Intialising the ANN

classifier = Sequential()
# Adding the input Layer And the first hidden layer

classifier.add(Dense(output_dim = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer

classifier.add(Dense(output_dim = 6, kernel_initializer='uniform', activation = 'relu'))
# Adding the output layer

classifier.add(Dense(output_dim = 1, kernel_initializer='uniform', activation = 'sigmoid'))
#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the training set

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
#Predicting the test set results

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)
y_pred
y_test
#Taking y_pred to boolean

y_tested = []

for i in range(len(y_test)):

    if y_test[i] == 1:

        y_tested.append([True])

    else:

        y_tested.append([False])
y_tested
# Making the confucion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_tested, y_pred)
cm
print("accuracy = ", (cm[0][0] + cm[1][1])*100/2000,"%")