# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/Churn_Modelling.csv')

dataset.head()
X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_geography = LabelEncoder()

X[:, 1] = labelencoder_geography.fit_transform(X[:, 1])

labelencoder_gender = LabelEncoder()

X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

#To avoid from dumy variable trap,we will remove one dumy variable column.

X = X[:,1:]

# Encoding the Dependent Variable.No need this because the dependent variable is in the 0-1 scale.

#labelencoder_y = LabelEncoder()

#y = labelencoder_y.fit_transform(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

#Notice that it is only transform not fit_transform!

X_test = sc.transform(X_test)
# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

import tensorflow as tf
print("Version: ", tf.__version__)
classifier = Sequential()
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)