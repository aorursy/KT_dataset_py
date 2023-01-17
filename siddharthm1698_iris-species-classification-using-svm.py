

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#importing the iris dataset. we have 4 features here which we are using for classifying our species. The output specifies 0 -> Iris-setosa 1 -> Iris-versicolor 2 -> Iris-virginica.



dataset = pd.read_csv('../input/Iris.csv')

X = dataset.iloc[:, [1,2,3,4]].values

y = dataset.iloc[:, 5].values

#splitting train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#scaling complete data for better predication

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
#using rbf by importing svc class from svm and using degree as 3 

from sklearn.svm import SVC

classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)

classifier.fit(X_train, y_train)
#predicting test data

y_pred = classifier.predict(X_test)

#getting the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm
#almost 96.66% accuracy of prediction on 20% of data on test