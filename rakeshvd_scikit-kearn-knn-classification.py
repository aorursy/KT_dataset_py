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
train_data = pd.read_csv('../input/train.csv',header = None)

train_labels = pd.read_csv('../input/trainLabels.csv',header = None)

test_data =  pd.read_csv('../input/test.csv',header = None)
train_data.head()
train_data.shape,test_data.shape,train_labels.shape
train_data.describe()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.30, random_state = 101)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.mixture import GaussianMixture



x_all = np.r_[train_data,test_data]

print('x_all shape :',x_all.shape)

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(x_all)

        bic.append(gmm.aic(x_all))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

best_gmm.fit(x_all)

gau_train = best_gmm.predict_proba(train_data)

gau_test = best_gmm.predict_proba(test_data)
knn = KNeighborsClassifier()

n_neighbors=[3,5,6,7,8,9,10]

param_grid = dict(n_neighbors=n_neighbors)



grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv = 10, n_jobs=-1,scoring='accuracy').fit(gau_train,train_labels.values.ravel())

knn_best = grid_search_knn.best_estimator_

print('KNN final best Score', grid_search_knn.best_score_)

print('KNN final best Params',grid_search_knn.best_params_)

print('KNN Accuracy',cross_val_score(knn_best,gau_train, train_labels.values.ravel(), cv=10).mean())

knn_best.fit(gau_train,train_labels.values.ravel())

pred  = knn_best.predict(gau_test)

knn_best_pred = pd.DataFrame(pred)



knn_best_pred.index += 1



knn_best_pred.columns = ['Solution']

knn_best_pred['Id'] = np.arange(1,knn_best_pred.shape[0]+1)

knn_best_pred = knn_best_pred[['Id', 'Solution']]



knn_best_pred.to_csv('Submission.csv',index=False)