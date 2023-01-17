import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
tr = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

te = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)



X_tr = np.array([tr[i][1] for i in range(2275)])

X_te = np.array([te[i][1] for i in range(976)])

y_tr = np.array([tr[i][0] for i in range(2275)])
tmp_X_tr=[]

tmp_X_te=[]



for l in range(2275):

    tmp1=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                tmp1.append(X_tr[l][i][j][k]/255.0)

    tmp_X_tr.append(tmp1)



for l in range(976):

    tmp1=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                tmp1.append(X_te[l][i][j][k]/255.0)

    tmp_X_te.append(tmp1)
cl = []

for i in range(2275):

    k=0

    for j in cl:

        if(j==tr[i][0]):

            k=1

    if(k==0):

        cl.append(tr[i][0])
from sklearn.decomposition import PCA



n_components = 500   

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(tmp_X_tr)



X_tr_pca = pca.transform(tmp_X_tr)

X_te_pca = pca.transform(tmp_X_te)



from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



svc = SVC(kernel='rbf', class_weight='balanced')

param_grid = {'C': [3,5,7,9],

              'gamma': [0.006,0.008,0.01]}

grid = GridSearchCV(svc, param_grid, cv=5,n_jobs=-1)



grid.fit(X_tr_pca, y_tr)

print(grid.best_params_)





model = grid.best_estimator_

y_pred = model.predict(X_te_pca)

y_pred
te.shape
tmp = np.array(range(976))

df =  pd.DataFrame({'ImageId' : tmp, 'Celebrity' : y_pred})



df.to_csv('ev4.csv', index = False)