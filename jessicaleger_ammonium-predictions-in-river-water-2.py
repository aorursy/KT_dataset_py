import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ammonium_train=pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/train.csv')

ammonium_train.head()
ammonium_test=pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/test.csv')

ammonium_test.head()
ammonium_train.drop(ammonium_train[['3','4','5','6','7']], axis=1, inplace=True)

ammonium_train.head()
ammonium_train.dropna(inplace=True)

ammonium_train.count()
from sklearn.model_selection import train_test_split
X=ammonium_train[['1','2']]

y=ammonium_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeRegressor
dtree=DecisionTreeRegressor()
dtree.fit(X_train, y_train)
ammonium_train_predictions=dtree.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error as mse
print("the mean squared error for the decision tree is", mse(y_test, ammonium_train_predictions))

print("the r2 score for the decision tree is", r2_score(y_test, ammonium_train_predictions))
from sklearn.tree import export_graphviz

import pydot
export_graphviz(dtree, out_file = 'tree.dot', rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')

from IPython.display import Image

Image('tree.png')
from sklearn.ensemble import AdaBoostRegressor
dtree_boosted= AdaBoostRegressor(dtree)
dtree_boosted.fit(X_train, y_train)

ammonium_train_predictions_boost=dtree_boosted.predict(X_test)
print("the mean squared error for the decision tree is", mse(y_test, ammonium_train_predictions_boost))

print("the r2 score for the decision tree is", r2_score(y_test, ammonium_train_predictions_boost))
sub_tree_5=dtree_boosted.estimators_[5]



export_graphviz(sub_tree_5, out_file = 'tree.dot', rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')

from IPython.display import Image

Image('tree.png')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf.fit(X_train, y_train)
ammonium_train_pred_rf=rf.predict(X_test)
print("the mean squared error for the random forest regression is", mse(y_test, ammonium_train_pred_rf))

print("the r2 score for the random forest regression is", r2_score(y_test, ammonium_train_pred_rf))
sub_tree_2=rf.estimators_[2]



export_graphviz(sub_tree_2, out_file = 'tree.dot', rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')

from IPython.display import Image

Image('tree.png')
rf_boosted= AdaBoostRegressor(rf)



rf_boosted.fit(X_train, y_train)

ammonium_train_pred_rf_b=rf_boosted.predict(X_test)
print("the mean squared error for the decision tree is", mse(y_test, ammonium_train_pred_rf_b))

print("the r2 score for the decision tree is", r2_score(y_test, ammonium_train_pred_rf_b))
sub_tree_3=rf_boosted.estimators_[3].estimators_[3]



export_graphviz(sub_tree_3, out_file = 'tree.dot', rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')

from IPython.display import Image

Image('tree.png')
from sklearn.svm import SVR
SVR_rbf=SVR(kernel='rbf')

SVR_rbf.fit(X_train, y_train)

SVR_pred=SVR_rbf.predict(X_test)
print("the mean squared error for SVR with rbf kernel is", mse(y_test, SVR_pred))

print("the r2 score for SVR with rbf kernel is", r2_score(y_test, SVR_pred))
SVR_lin=SVR(kernel='linear')

SVR_lin.fit(X_train, y_train)

SVR_lin_pred=SVR_lin.predict(X_test)
print("the mean squared error for SVR with linear kernel is", mse(y_test, SVR_lin_pred))

print("the r2 score for SVR with linear kernel is", r2_score(y_test, SVR_lin_pred))
SVR_poly=SVR(kernel='poly')

SVR_poly.fit(X_train, y_train)

SVR_poly_pred=SVR_poly.predict(X_test)
print("the mean squared error for SVR with polynomial kernel is", mse(y_test, SVR_poly_pred))

print("the r2 score for SVR with polynomial kernel is", r2_score(y_test, SVR_poly_pred))
import timeit
benchmark_results = pd.DataFrame(columns=["Code", "Trial 1 (ms)", "Trial 2 (ms)", "Trial 3 (ms)", "Mean (ms)"])

benchmark_codes = ['dtree', 'dtree_boosted','rf', 'rf_boosted', 'SVR_rbf', 'SVR_lin', 'SVR_poly']



for index, code in enumerate(benchmark_codes):

    row = [code]

    results = timeit.repeat(f'{code}.predict(X_test)', f'from __main__ import {",".join(globals())}', repeat=3, number=10)

    row.extend(results)

    row.append(sum(results)/len(results))

    benchmark_results.loc[index] = row



benchmark_results.round(decimals=4)