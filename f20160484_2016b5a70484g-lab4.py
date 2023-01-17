import warnings



import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.decomposition import PCA

import sklearn

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline



train = np.load('/kaggle/input/eval-lab-4-f464/train.npy', allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy', allow_pickle=True)
(train.shape, test.shape)
X_train = np.array([train[i][1] for i in range(2275)])

X_test = np.array([test[i][1] for i in range(976)])

y_train = np.array([train[i][0] for i in range(2275)])
X_train_new=[]

X_test_new=[]

for x in range(2275):

    temp=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                temp.append(X_train[x][i][j][k]/255.0)

    X_train_new.append(temp)
for x in range(976):

    temp=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                temp.append(X_test[x][i][j][k]/255.0)

    X_test_new.append(temp)

 
clusters = []

for i in range(2275):

    flag = 0

    for j in clusters:

        if(j == train[i][0]):

            flag = 1

    if(flag == 0):

        clusters.append(train[i][0])

 

len(clusters)
 

n_components = 120

pca = PCA(n_components = n_components, svd_solver='randomized', whiten=True, random_state=42).fit(X_train_new)

 

X_train_pca = pca.transform(X_train_new)

X_test_pca = pca.transform(X_test_new)


 

svc = SVC( kernel='rbf', class_weight='balanced' )

param_grid = {'C': [3,5,7,9],

              'gamma': [0.006,0.008,0.01]}

grid = GridSearchCV(svc, param_grid, cv=5)

 

grid.fit(X_train_pca, y_train)

print(grid.best_params_)
model = grid.best_estimator_

y_pred = model.predict(X_test_pca)
test.shape
imgid = np.array(range(976))

sub = pd.DataFrame({'ImageId' : imgid, 'Celebrity' : y_pred})

sub.shape
sub.to_csv('submission1.csv',index=False)

    