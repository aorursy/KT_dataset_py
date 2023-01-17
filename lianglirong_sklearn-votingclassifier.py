import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/pima_data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8]
num_tree = 100
max_features = 7
seed = 2

models = []
model_logistic =  LogisticRegression()
models.append(("logistic",model_logistic))
model_cart = DecisionTreeClassifier()
models.append(("cart",model_cart))
model_svc = SVC()
models.append(("svc",model_svc))

model = VotingClassifier(models)
kfold = KFold(n_splits=10,random_state=seed)
result = cross_val_score(model,X,Y,cv=kfold)

print("result:",result.mean())
