import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

import seaborn as sns
train = np.load('train.npy',allow_pickle=True)

test = np.load('test.npy',allow_pickle=True)
X_train = np.array([train[i][1] for i in range(2275)])

X_test = np.array([test[i][1] for i in range(976)])

y_train=np.array([train[i][0] for i in range(2275)])
new_X_train=[]

new_X_test=[]

for z in range(len(train)):

    temp=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                temp.append(X_train[z][i][j][k]/255.0)

    new_X_train.append(temp)

    

for z in range(len(test)):

    temp=[]

    for i in range(50):

        for j in range(50):

            for k in range(3):

                temp.append(X_test[z][i][j][k]/255.0)

    new_X_test.append(temp)
clusters = []

for i in range(len(train)):

    f=0

    for j in clusters:

        if(j==train[i][0]):

            f=1

    if(f==0):

        clusters.append(train[i][0])



len(clusters)
from sklearn.decomposition import PCA



n_components = 90

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False, random_state=0).fit(new_X_train)



X_train_pca = pca.transform(new_X_train)

X_test_pca = pca.transform(new_X_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



svc = SVC(kernel='rbf', class_weight='balanced')

param_grid = {'C': [1, 5, 10, 15],

              'gamma': [0.005,0.01,0.02,0.03]}

grid = GridSearchCV(svc, param_grid, cv=5)

    

grid.fit(X_train_pca, y_train)

print(grid.best_params_)
model = grid.best_estimator_

y_pred = model.predict(X_test_pca)

y_pred
out = [[test[i][0],y_pred[i]] for i in range(len(test))]

out_df = pd.DataFrame(data=out,columns=['ImageId','Celebrity'])

out_df.to_csv(r'out_2_2.csv',index=False)