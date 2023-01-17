# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data visulization and Analysis



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.dtypes
df.isna().sum()
df.shape
df.rename(columns={'sales': 'department'}, inplace=True)
df.department.unique()
df.salary.unique()
# convert the categorical columns to numbers by converting them to dummy variables
df_final = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)
df_final
from sklearn.model_selection import train_test_split
# We will predict left column



# input features

X = df_final.drop(['left'], axis=1).values



# output

y = df_final['left'].values
X
# Spliting data into training and testing (70% training and 30% testing)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train
# To scale the training set and the test set

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense
# Sequential to initialize a linear stack of layers

# Since this is a classification problem, we'll create a classifier variable

classifier = Sequential()
# adding layers to your network

classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim = 18))
classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# To evaluate how well the model performed on the predictions, you will next use a confusion matrix.
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))
new_pred = (new_pred > 0.6)
new_pred
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
def make_classifier():

    classifier = Sequential()

    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))

    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

    classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

    return classifier
classifier = KerasClassifier(build_fn = make_classifier, batch_size = 10, nb_epoch = 1)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()

mean
variance = accuracies.var()

variance
from keras.layers import Dropout



classifier = Sequential()

classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim = 18))

classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])



from sklearn.model_selection import GridSearchCV



def make_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim = 18))

    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

    return classifier
classifier = KerasClassifier(build_fn = make_classifier)
params = {

    'batch_size':[20,35],

    'epochs':[2,3],

    'optimizer':['adam','rmsprop']

}
grid_search = GridSearchCV(estimator = classifier, param_grid = params, scoring = "accuracy", cv = 2)
grid_search = grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_

best_accuracy = grid_search.best_score_
best_param
best_accuracy
# used Keras to build an artificial neural network that predicts the probability that an employee will leave a company