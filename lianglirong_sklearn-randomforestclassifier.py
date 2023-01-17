import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("../input/pima_data.csv",names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]
num_tree = 100
seed = 2
max_features = 3
model = RandomForestClassifier(n_estimators=num_tree,random_state=seed,max_features=max_features)
kfold = KFold(n_splits=10,random_state=seed)
result = cross_val_score(model,X,Y,cv=kfold)
print("result:",result.mean())
