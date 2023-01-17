import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/data.csv")
data.head(5)
data.columns
data.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
data.head(5)
data.dtypes
data = data.values
x = data[ : , 1:]
y = data[ : , 0]
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, shuffle = True)
print(x_train.shape,
     y_train.shape,
     x_test.shape,
     y_test.shape)
x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(x_train, y_train, test_size = 0.15, shuffle = True)
print(x_train_val.shape,
     y_train_val.shape,
     x_test_val.shape,
     y_test_val.shape)
clf = XGBClassifier()
clf.fit(x_train_val, y_train_val)
pred = clf.predict(x_test_val)
print(pred, y_test_val, sep = '\n')
print("Accuracy of model is: ", accuracy_score(y_test_val, pred))
n_estimators = list(range(50, 400, 50))
max_depth = list(range(1, 11, 2))
learning_rate = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
params_grid = dict(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate)
print(params_grid)
grid_search = GridSearchCV(clf, params_grid, scoring="neg_log_loss", n_jobs=-1, cv = 10, verbose = 3)
grid_result = grid_search.fit(x_train_val, y_train_val)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
clf1 = XGBClassifier(learing_rate = 0.1, max_depth = 1, n_estimators = 250)
clf1.fit(x_train_val, y_train_val)
pred = clf.predict(x_test)
print("Accuracy of model is: ", accuracy_score(y_test, pred))
