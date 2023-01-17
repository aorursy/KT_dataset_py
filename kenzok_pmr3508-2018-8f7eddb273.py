import pandas as pd
import sklearn
household = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="NaN")
household
household.isnull().sum(axis = 0).sort_values(ascending = False)
Dhousehold = household.drop(columns = ['rez_esc', 'v18q1', 'v2a1', 'Id'])
nDhousehold = Dhousehold.dropna()
nDhousehold.shape
testhousehold = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="NaN")
Dtesthousehold = testhousehold.drop(columns = ['rez_esc', 'v18q1', 'v2a1', 'Id'])
nDtesthousehold = Dtesthousehold.dropna()
nDtesthousehold.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import numpy as np
numnDhousehold = nDhousehold.apply(preprocessing.LabelEncoder().fit_transform)
numnDtesthousehold = nDtesthousehold.apply(preprocessing.LabelEncoder().fit_transform)
Xhousehold = numnDhousehold.iloc[:,0:138]
Yhousehold = nDhousehold.Target
knn = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
scores
np.average(scores)
Xhousehold = Xhousehold.drop(columns = ['age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'])
Xhousehold.shape
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
scores
np.average(scores)
Xhousehold = numnDhousehold.iloc[:,0:138]
knn = KNeighborsClassifier(n_neighbors = 10)
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
scores
np.average(scores)
knn = KNeighborsClassifier(n_neighbors = 20)
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
np.average(scores)
knn = KNeighborsClassifier(n_neighbors = 30)
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
np.average(scores)
knn = KNeighborsClassifier(n_neighbors = 40)
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
np.average(scores)
knn = KNeighborsClassifier(n_neighbors = 45)
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv = 10)
np.average(scores)
knn = KNeighborsClassifier(n_neighbors = 40)
knn.fit(Xhousehold, Yhousehold)
Xtesthousehold = numnDtesthousehold.iloc[:,0:138]
Ytesthousehold = knn.predict(Xtesthousehold)
