import pandas as pd
import sklearn
import numpy as np
from sklearn.grid_search import GridSearchCV
import os
os.listdir("../input/")
adult_train = pd.read_csv("../input/train_data.csv",na_values="?")
adult_test = pd.read_csv("../input/test_data.csv",na_values="?")
adult_train.head()
adult_train = adult_train.dropna()
adult_train
Xadult = adult_train[["age","education.num","capital.gain","hours.per.week","capital.loss"]]
Yadult = adult_train.income
Xadult_test = adult_test[["age","education.num","capital.gain","hours.per.week","capital.loss"]]
Xadult.shape
Xadult_test.shape
#from sklearn import preprocessing
#Xadult = Xadult.apply(preprocessing.LabelEncoder().fit_transform)
#Xadult_test = Xadult_test.apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn = KNeighborsClassifier(n_neighbors=20)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(Xadult_test)
df = pd.DataFrame(YtestPred)
df.to_csv("submit2")
Xadult = adult_train[["age","occupation","relationship","education.num","capital.loss","capital.gain","hours.per.week","workclass","sex","race"]]
Xadult = pd.get_dummies(Xadult)

Xadult.head()
Xadult = Xadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week","sex_Female","race_White","workclass_Private","occupation_Adm-clerical","occupation_Prof-specialty","relationship_Wife","relationship_Husband"]]
adult_test = pd.get_dummies(adult_test)
Xadult_test = adult_test[["age","education.num","capital.gain", "capital.loss", "hours.per.week","sex_Female","race_White","workclass_Private","occupation_Adm-clerical","occupation_Prof-specialty","relationship_Wife","relationship_Husband"]]
Xadult_test.shape
knn = KNeighborsClassifier(n_neighbors=20, leaf_size=27, p=1)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(Xadult_test)
df = pd.DataFrame(YtestPred)
df.to_csv("submit6")