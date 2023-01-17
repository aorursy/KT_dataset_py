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
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 784, activation = 'tanh', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(units = 128, activation = 'tanh'))
classifier.add(Dense(units = 128, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 64, activation = 'tanh'))
classifier.add(Dense(units = 64, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 32, activation = 'tanh'))
classifier.add(Dense(units = 32, activation = 'tanh'))
classifier.add(Dropout(0.05))
# Adding the output layer
classifier.add(Dense(units = 10, activation = 'softmax'))
# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, 
                         validation_data = (X_test, y_test), 
                         batch_size = 28, 
                         epochs = 25)

model_acc = classifier.evaluate(X_test, y_test)
print(" Model Accuracy is : {0:.1f}%".format(model_acc[1]*100))

test_dataset = pd.read_csv('../input/test.csv')
test = test_dataset.iloc[:,:]
# Prediction
test_pred = classifier.predict(test)
# Mark probability score > 0.5 as Predicted Label, axis = 1 means insert column-wise 
results = test_pred.argmax(axis=1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("solution.csv",index=False)