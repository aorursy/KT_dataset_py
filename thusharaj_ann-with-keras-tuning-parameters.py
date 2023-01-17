# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Importing the data file from "../input/" directory.

data = pd.read_csv('../input/Breast_cancer_data.csv')

data.info()

X = data.iloc[:,0:5].values

y = data.iloc[:,5].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Import Keras

from keras.wrappers.scikit_learn import KerasClassifier

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import GridSearchCV

# build classifier function 

def build_classifier_best(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

# find the best parameters 

classifier = KerasClassifier(build_fn = build_classifier_best)

parameters = {'batch_size':[25,32],

              'epochs':[100,500],

              'optimizer':['adam','rmsprop']}



grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
