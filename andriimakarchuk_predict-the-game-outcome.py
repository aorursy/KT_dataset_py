import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression
data = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
X = data.drop( labels=["blueWins"], axis=1 )

y = data["blueWins"]
X.corr()
regModel = LinearRegression()

regModel.fit(X, y)
print( "R^2="+str(regModel.score(X, y)) )
corrTable = X.corr()

cols = X.columns
newCols = []

for i in cols:

    for j in cols:

        if( i!=j and abs(corrTable[i][j])<0.01 ):

            newCols.append(j)

newCols = list( pd.Series(newCols).unique() )

print( len(newCols) )
X = data[newCols]

regModel = LinearRegression()

regModel.fit(X, y)

print( "R^2="+str(regModel.score(X, y)) )
from sklearn import tree

treeRegressor = tree.DecisionTreeRegressor(

    min_samples_leaf=100,

    max_leaf_nodes=2

)

treeRegressor.fit(X, y)
print( "R^2="+str(treeRegressor.score(X, y)) )
from sklearn import tree

treeClassifier = tree.DecisionTreeClassifier(

    min_samples_leaf=100,

    max_leaf_nodes=2

)

treeClassifier.fit(X, y)

print( "R^2="+str(treeClassifier.score(X, y)) )
tree.plot_tree(treeClassifier)
from sklearn.linear_model import LogisticRegression

logRegModel = LogisticRegression()

logRegModel.fit(X, y)

print( "R^2="+str(logRegModel.score(X, y)) )