import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv").dropna(how="all").drop(labels="Unnamed: 32", axis=1)
X = data.drop( labels="diagnosis", axis=1 )

y = data["diagnosis"].to_list()

for i in range(len(y)):

    if( y[i]=="B" ):

        y[i]=1

    else:

        y[i]=0

y = pd.Series(y)



print( y )
print( f_regression(X, y) )
reg_model = LinearRegression()

reg_model.fit(X, y)
print( reg_model.score(X, y) )
print( len(X.columns) )
#print of coeffitients of linear model

print( reg_model.coef_ )