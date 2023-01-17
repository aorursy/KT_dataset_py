import pandas as pd
import sklearn
import numpy as np
import os
from sklearn.grid_search import GridSearchCV
train = pd.read_csv("../input/train.csv",na_values="NaN")
test = pd.read_csv("../input/test.csv",na_values="NaN")
train.head()
Idfin = test.Id
Y_train = train.Target
X_train = train
for x in ['Id','Target','v18q1','r4h1','r4m1','r4t1','r4t2','r4t3','idhogar','dependency','edjefe','edjefa','male','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned']:
    X_train.pop(x)
X_test = test
for x in  ['Id','v18q1','r4h1','r4m1','r4t1','r4t2','r4t3','idhogar','dependency','edjefe','edjefa','male','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned']:
    X_test.pop(x)
from sklearn import preprocessing
X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)
X_test = X_test.apply(preprocessing.LabelEncoder().fit_transform)
X_train.dropna()
Y_train.dropna()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, X_train, Y_train, cv=10)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)

scaler1 = StandardScaler()
# Fit on training set only.
scaler1.fit(X_test) 
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, X_train, Y_train, cv=10)
scores
knn.fit(X_train,Y_train)
YtestPred = knn.predict(X_test)
output = pd.DataFrame(Idfin)
output["Target"] = YtestPred
df = pd.DataFrame(output)
df.to_csv("CR10.csv", index =False)
