# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv('/kaggle/input/Advertising_data.csv')

X = dataset.iloc[:,[2,3]].values

y = dataset.iloc[:,4].values
dataset.head()
# train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Fitting kernel SVM to training data

from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)

classifier.fit(X_train, y_train)
# Predict

y_pred = classifier.predict(X_test)
# metrics before GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Classification Report:")

print(classification_report(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:")

print(accuracy_score(y_test, y_pred))
# Applying GridSearchCV

from sklearn.model_selection import GridSearchCV

parameters = [{'C':[1, 10, 100, 1000], 'kernel':['linear']},

             {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,

                          param_grid = parameters,

                          scoring = 'accuracy',

                          cv=10,

                          n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
grid_search.best_score_
grid_search.best_params_
classifier = SVC(kernel = 'rbf', gamma=0.7)

classifier.fit(X_train, y_train)
# Predict

y_pred = classifier.predict(X_test)
# metrics after GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Classification Report:")

print(classification_report(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:")

print(accuracy_score(y_test, y_pred))