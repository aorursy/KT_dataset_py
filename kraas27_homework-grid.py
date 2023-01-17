import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

from sklearn.datasets import load_boston

from sklearn.model_selection import GridSearchCV
df = load_boston()
X = pd.DataFrame(df.data, columns=df.feature_names)

y = pd.DataFrame(df.target)

y.columns = ['TARGET']

data_all = pd.concat([X, y], axis=1)
data_all.head(3)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2) 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_valid = sc.transform(X_valid)
X_train.shape[0] == y_train.shape[0]
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
max_depth = list(range(3, 31, 3))

min_samples_leaf = list(range(5, 31, 5))

params_grid = {'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf}

rf_grid = GridSearchCV(rf, params_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
print (rf_grid.best_score_)

print (rf_grid.best_params_)

rf_grit_best = rf_grid.best_estimator_
from sklearn.metrics import mean_squared_error

y_pred = rf_grit_best.predict(X_valid)

mse_rf = mean_squared_error(y_valid, y_pred)

mse_rf
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
max_depth = [3, 4, 5, 7, 9, 12, 15] 

min_samples_leaf = [1, 2, 3, 5, 7, 10, 15]

params_grid = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

tree_grid = GridSearchCV(tree_reg, params_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
tree_grid.fit(X_train, y_train)
print (tree_grid.best_score_)

print (tree_grid.best_params_)

tree_grit_best = tree_grid.best_estimator_
y_pred = tree_grit_best.predict(X_valid)

mse_tree = mean_squared_error(y_valid, y_pred)

mse_tree
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_jobs=-1)
n_neighbours = [2, 3, 4, 5, 6, 7, 8, 9, 10]

weight = ['uniform', 'distance']

params_grid = {'n_neighbors': n_neighbours, 'weights': weight}

knn_grid = GridSearchCV(knn_reg, params_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
print (knn_grid.best_score_)

print (knn_grid.best_params_)

knn_grit_best = knn_grid.best_estimator_
y_pred = knn_grit_best.predict(X_valid)

mse_knn = mean_squared_error(y_valid, y_pred)

mse_knn
from sklearn.linear_model import Lasso

lasso_reg = Lasso()
alpha = [1, 0.1, 0.05, 0.01, 0.001]

params_grid = {'alpha': alpha}

lasso_grid = GridSearchCV(lasso_reg, params_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
print (lasso_grid.best_score_)

print (lasso_grid.best_params_)

lasso_grit_best = lasso_grid.best_estimator_
y_pred = lasso_grit_best.predict(X_valid)

mse_lasso = mean_squared_error(y_valid, y_pred)

mse_lasso
from sklearn.svm import SVR

svm = SVR()
kernel = ['linear', 'poly', 'rbf', 'sigmoid']

C = [0.1, 1, 2, 3, 4, 5, 8, 10, 20, 40, 60, 100]

params_grid = {'kernel': kernel, 'C': C}

svm_grid = GridSearchCV(svm, params_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
svm_grid.fit(X_train, y_train)
print (svm_grid.best_score_)

print (svm_grid.best_params_)

svm_grit_best = svm_grid.best_estimator_
y_pred = svm_grit_best.predict(X_valid)

mse_svm = mean_squared_error(y_valid, y_pred)

mse_svm
print ('mse_rf ', mse_rf)

print ('mse_tree ', mse_tree)

print ('mse_knn ', mse_knn)

print ('mse_lasso ', mse_lasso)

print ('mse_svm ', mse_svm)