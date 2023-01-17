import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression
data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv").dropna()

data.head()
data = data.drop( labels="date", axis=1 )

data.head()
X = data.drop(labels="price", axis=1)

y = data["price"]



X.corr()
print( f_regression(X, y) )
regModel = LinearRegression()

regModel.fit(X, y)

print( regModel.score(X, y) )
from sklearn import tree

classModel = tree.DecisionTreeClassifier(

    min_samples_leaf = 100

)

classModel.fit(X, y)

print( classModel.score(X, y) )
#building quadratic regression model

newX = []



for i in X.index:

    newX.append( list(X.loc[i])+list(np.power(X.loc[i], 2)) )



regModel.fit(newX, y)

print( regModel.score(newX, y) )
newX = []



for i in X.index:

    newX.append( list(X.loc[i])+list(np.cbrt(X.loc[i])) )



regModel.fit(newX, y)

print( regModel.score(newX, y) )
newX = []



for i in X.index:

    newX.append( list(X.loc[i])+list( np.cbrt(X.loc[i]) )+list( np.cbrt(np.cbrt(X.loc[i])) ) )



regModel.fit(newX, y)

print( regModel.score(newX, y) )