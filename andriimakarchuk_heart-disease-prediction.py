import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression
data = pd.read_csv("../input/heart-disease/heart.csv").dropna()
X = data.drop( labels="target", axis=1 )

y = data["target"]
f_regression(X, y)
X = X.drop( labels=X.columns[3:5], axis=1 )

X.corr()
regModel = LinearRegression().fit(X, y)
print( "R^2="+str(regModel.score(X, y)) )
newX = []

for i in X.index:

    newX.append( X.loc[i].to_list()+np.power(X.loc[i].to_list(), 2) )

regModel.fit(newX, y)

print( "R^2="+str(regModel.score(X, y)) )
from sklearn import tree

treeRegressor = tree.DecisionTreeRegressor(

    min_samples_leaf=10

).fit(X, y)

treeClassifier = tree.DecisionTreeClassifier(

    min_samples_leaf=10,

    max_leaf_nodes=2

).fit(X, y)
print( "R^2 for tree regression: "+str(treeRegressor.score(X, y)) )

print( "R^2 for tree classification: "+str(treeClassifier.score(X, y)) )
tree.plot_tree(treeClassifier)