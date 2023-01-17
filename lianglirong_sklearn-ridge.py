import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

import os
print(os.listdir("../input"))
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("../input/housing.csv",names=names,delim_whitespace=True)

array = data.values
X = array[:,0:13]
Y = array[:,13]
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=10,random_state=2)
model = Ridge()
result = cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
print("result:",result.mean())
