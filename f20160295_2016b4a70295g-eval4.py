import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

train_np = np.load('train.npy')

test_np = np.load('test.npy')
X_train = np.array([train[i][1].flatten() for i in range(len(train_np))])

X_test = np.array([test[i][1].flatten() for i in range(len(test_np))])

y_train=np.array([train[i][0] for i in range(len(train_np))])
X_train = X_train/255

X_test = X_test/255
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PC
n_components = 100

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
svc = SVC(kernel='rbf', class_weight='balanced')

# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],

#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}

param_grid = {

        'C': [2,4,7,12, 25, 75, 100],

      'gamma': [0.004,0.008,0.016]

}



grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=3)

grid.fit(X_train_pca, y_train)

print(grid.best_score_)     
y_pred = grid.best_estimator_.predict(X_test_pca)

y_pred
df_sub = pd.DataFrame({

    'ImageId':test[:,0],

    'Celebrity':y_pred

})

df_sub.to_csv("sub7.csv", index=False)