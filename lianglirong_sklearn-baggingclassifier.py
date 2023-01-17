import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import os
print(os.listdir("../input"))
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("../input/pima_data.csv",names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]
cart = DecisionTreeClassifier()
num_tree = 10
seed = 2

model = BaggingClassifier(cart,n_estimators=num_tree,random_state=seed)
kfold = KFold(n_splits=10,random_state=seed)
result = cross_val_score(model,X,Y,cv=kfold)
print("result:",result.mean())
