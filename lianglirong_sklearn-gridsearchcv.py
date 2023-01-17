import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/pima_data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8]
seed = 2
model = Ridge()
param_grid = {'alpha':[1,0.1,0.01,0.001,0]}
grid = GridSearchCV(model,param_grid=param_grid)
grid.fit(X,Y)
print("best score:",grid.best_score_)
print("best alpha:",grid.best_estimator_.alpha)
