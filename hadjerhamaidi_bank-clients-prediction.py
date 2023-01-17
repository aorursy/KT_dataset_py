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
file = '../input/Churn_Modelling.csv'
#1- importing necessary libs
import matplotlib.pyplot as plt
import pandas as pd
#2- load dataset into Pandas DataFrame
dataset = pd.read_csv(file)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#3- Encode Categorical dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
#4- Split dataset into train ans test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#5- Features Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#6- import required packages for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#7 Building the ANN with Stochastic Gradient Descent
#7-0 Initializing the ANN
classifier = Sequential()
#7-1 Adding the input layer and the 1st hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p=0.1))
#7-2 Adding the 2nd hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p=0.1))
#7-3 Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#8- Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
#9- Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
#10- Predicting the test results
y_pred = classifier.predict(X_test)
print(y_train)
y_pred = (y_pred > 0.5)
print(y_pred)
#11- Making the confurion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
#12- Predicting if this following client will leave the bank or not 
#     Geography: Germany
#     Credit Score: 750
#     Gender: Male
#     Age: 30 years old
#     Tenure: 2 years
#     Balance: $60000
#     Number of Products: 2
#    Does this customer have a credit card ? Yes
#    Is this customer an Active Member: Yes
#     Estimated Salary: $100000
client_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 300, 1, 30, 2, 60000, 2, 1, 1, 100000]])))
client_prediction = (client_prediction > 0.5)
print(client_prediction)
#13-  Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
mean = accuracies.mean()
variance = accuracies.std()
#14- Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
             'nb_epoch' : [100, 500],
             'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

