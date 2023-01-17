# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



X = np.array([[0, 2.10, 1.45],

              [2, 1.18, 1.33],

              [0, 1.22, 1.27],

              [1, 1.32, 1.97],

              [1, -0.21, -1.19]])



X_nan = np.array([[np.nan, 0.87, 1.31],

                   [np.nan, 0.37, 1.91],

                   [np.nan, 0.54, 1.27],

                   [np.nan, -0.67, -0.22]])



clf = KNeighborsClassifier(3, weights='distance')

trained_model = clf.fit(X[:,1:], X[:,0])



#predict missing value's class

imputed_values = trained_model.predict(X_nan[:,1:])

print("\nImputed value"); print(imputed_values)



#join column of predicted class

X_imputed_values = np.hstack((imputed_values.reshape(-1,1), X_nan[:,1:]))

print("\nX Imputed value"); print(X_imputed_values)



print("\nJoin Feature Matrix"); print(np.vstack((X_imputed_values, X)))