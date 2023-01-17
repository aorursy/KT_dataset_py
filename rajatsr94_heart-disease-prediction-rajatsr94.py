# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

dataset = pd.read_csv("../input/heart.csv")



# Getting to know about abour dataset

print(dataset.head())

print(dataset.shape)

print(dataset.columns)

print(dataset.isnull().sum()) #checking if our dataset has any missing values

print(dataset.describe())

corr = dataset.corr()

sns.heatmap(corr, annot = True)



# Splitting the Dependent & Independent variables from dataset

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values



# Splitting the data into Training set & Testing set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Naive Bayes classification technique

from sklearn.naive_bayes import GaussianNB

classifier1 = GaussianNB()

classifier1.fit(X_train, y_train)

y_pred1 = classifier1.predict(X_test)



# making Confusion Matrix for NB classifier

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred1)

print("Accuracy of NB: '{}%'".format(cm1.diagonal().sum() * 100 / cm1.sum()))
# Random Forest classification technique

from sklearn.ensemble import RandomForestClassifier

classifier2 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)

classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)



# making Confusion Matrix for RF classifier

cm2 = confusion_matrix(y_test, y_pred2)

print(cm2)

print("Accuracy of RF: '{}%'".format(cm2.diagonal().sum() * 100 / cm2.sum()))
print(classifier2.feature_importances_)

print(classifier2.get_params)



# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{

    'n_estimators' : [50, 100, 150, 200],

    'min_samples_split' : [2, 4, 6],

    'min_samples_leaf' : [1, 3, 5, 7],

    'max_depth' : [5, 10, 15, None],

    'max_features' : ['auto', 'sqrt']}]

grid_search = GridSearchCV(estimator = classifier2,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
# Applying Random Forest classification after getting best parameters through Grid Search

from sklearn.ensemble import RandomForestClassifier

classifier3 = RandomForestClassifier(n_estimators = 100, max_features = 'auto', min_samples_leaf = 1,

                                     min_samples_split = 2, max_depth = 5, criterion = 'entropy', random_state = 0)

classifier3.fit(X_train, y_train)

y_pred3 = classifier3.predict(X_test)



# making Confusion Matrix for updated RF classifier

cm3 = confusion_matrix(y_test, y_pred3)

print(cm3)

print("Accuracy of updated RF: '{}%'".format(cm3.diagonal().sum() * 100 / cm3.sum()))
# now applying K-fold Cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier3, X = X_train, y = y_train, cv = 5)

print(accuracies.mean())

print(accuracies.std())