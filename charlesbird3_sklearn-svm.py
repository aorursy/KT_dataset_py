# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import time 

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



%matplotlib inline



train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")

train_data = train.values

print("Train Data Shape is: ",train_data.shape)

train.head()



# Any results you write to the current directory are saved as output.
test_data = test.values

print("Test Data Shape is: ",test_data.shape)

test.head()
X_train = train_data[:, 1:]

y_train = train_data[:, 0]

print('Train Data shape: ', X_train.shape)

print('Label shape: ', y_train.shape)
X_test = test_data[:]

print('Test Data shape: ', X_test.shape)
sc = StandardScaler()

sc.fit(X_train)

X_std = sc.transform(X_train)

print('Standard Data shape:', X_std.shape)

X_std[:5, :]
X_test_std = sc.transform(X_test)

print('Standard Test Data shape:', X_test_std.shape)

X_test_std[:5, :]
def find_SVC_hyper_param(X, y):

    param_grid = [

        {

            'C': [100, 10, 1],

            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            # 'degree': [3, 4, 5],

            'gamma': ['auto', 0.1, 0.01, 0.001],

            # 'tol': [1e-3, 1e-4, 1e-5],

            # 'random_state': [0, 100, 200, 500]

        },

    ]

    grid_search = GridSearchCV(SVC(), param_grid, n_jobs=-1, verbose=1, cv=5)

    grid_search.fit(X, y)

    print('best_SVC_score', grid_search.best_score_)

    print('best_SVC_param', grid_search.best_params_)

    return grid_search.best_estimator_
pca = PCA(n_components=0.95)

pca.fit(X_std)

X_std_pca = pca.transform(X_std)

print('Standard PCA Data shape: ', X_std_pca.shape)
X_test_std_pca = pca.transform(X_test_std)

print('Standard PCA Test Data shape: ', X_test_std_pca.shape)
start_time = time.time()

best_svc = find_SVC_hyper_param(X_std_pca, y_train)

elapsed_time = time.time() - start_time

print('best_SVC_estimator', best_svc)

print("Time consumed to find hyper param: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
y_predict = best_svc.predict(X_test_std_pca)
submissions=pd.DataFrame({"ImageId": list(range(1,len(y_predict)+1)),

                         "Label": y_predict})

submissions.to_csv("PCA95-SVM.csv", index=False)