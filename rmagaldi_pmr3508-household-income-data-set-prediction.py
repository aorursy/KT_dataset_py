# Importing the libraries

import numpy as np
import pandas as pd
import sklearn
import matplotlib as plt
# Uploading the data set

raw_train = pd.read_csv("../input/household-income-data-set/train.csv", sep=r'\s*,\s*', engine='python', na_values="?")
raw_train.head()
raw_train.shape
# Replacing string values for numeric ones

raw_train_ = raw_train.replace(np.nan,'0', regex=True)
raw_train__ = raw_train_.replace("no",'0', regex=False)
rraw_train = raw_train__.replace("yes",'1', regex=False)
rraw_train.head(20)
# Some data visualization

rraw_train.Target.value_counts().plot(kind="pie", autopct='%1.1f%%')
rraw_train.dependency.value_counts().plot(kind="bar")
rraw_train.rooms.value_counts().plot(kind="bar")
rraw_train.tamviv.value_counts().plot(kind="bar")
rraw_train.corr().Target
# Selecting most correlated values in order to improve the prediction

lista = []
lista_scores = []
lista_atrib = []
for i in range(len(rraw_train.corr().Target)-1):
    if abs(rraw_train.corr().Target[i])>0.15:
        lista.append((rraw_train.corr().Target.index[i], rraw_train.corr().Target[i]))
        lista_scores.append(rraw_train.corr().Target[i])
        lista_atrib.append(rraw_train.corr().Target.index[i])
lista_atrib
lista_scores
# Defining actual training data

Xtrain = rraw_train[lista_atrib]
Ytrain = rraw_train.Target
# Uploading test data

raw_test = pd.read_csv("../input/household-income-data-set/test.csv", sep=r'\s*,\s*', engine='python', na_values="?")
raw_test.head(20)
raw_test.shape
# Replacing some useless values for useful ones

raw_test_ = raw_test.replace(np.nan,'0', regex=True)
raw_test__ = raw_test_.replace("no",'0', regex=False)
rraw_test = raw_test__.replace("yes",'1', regex=False)
rraw_test.head(20)
# Defining actual testing data

test = rraw_test[lista_atrib]
test.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# Obtaining the best k for the kNN classifier

lista_knn = []
for i in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10).mean()
    lista_knn.append(scores)
lista_knn
max(lista_knn)
# Since the best result came with k=25, we ought to implement it

def_knn = KNeighborsClassifier(n_neighbors=25)
def_knn.fit(Xtrain,Ytrain)
YtestPred = def_knn.predict(test)
YtestPred.shape
predict = pd.DataFrame(rraw_test.Id)
predict["Target"] = YtestPred
predict
predict.to_csv("prediction.csv", index=False)