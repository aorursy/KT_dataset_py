
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
print(os.listdir("../input"))

data = pd.read_csv("../input/pima_data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10,random_state=2)
model = LinearDiscriminantAnalysis()
result = cross_val_score(model,X,Y,cv=kfold)
print("result:",result.mean())
