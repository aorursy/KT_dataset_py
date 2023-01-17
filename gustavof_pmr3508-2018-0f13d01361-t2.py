import pandas as pd
import sklearn
import numpy as nm
filetest = "../input/test.csv"
hhtest = pd.read_csv(filetest,
        na_values="?")
hhtest.shape
hhtest.head()
hhtrain = pd.read_csv("../input/train.csv",
        na_values="?")                   
hhtrain.shape
clean = hhtrain.dropna()
hhtrain.head()
Xhh = hhtrain[["r4m1", "r4t1", "paredblolad", "paredmad", "tipovivi2", "tipovivi3", "qmobilephone", "lugar4", "SQBedjefe", "SQBhogar_nin", "SQBdependency"]]
Yhh = hhtrain.Target
Xtest = hhtest[["r4m1", "r4t1", "paredblolad", "paredmad", "tipovivi2", "tipovivi3", "qmobilephone", "lugar4", "SQBedjefe", "SQBhogar_nin", "SQBdependency"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
x = 5
for i in range(0,40):
    knn = KNeighborsClassifier(n_neighbors=(x))
    scores = cross_val_score(knn, Xhh, Yhh, cv=10)
    print(x)
    print(nm.mean(scores))
    x = x + 5
knn = KNeighborsClassifier(n_neighbors=(40))
scores = cross_val_score(knn, Xhh, Yhh, cv=10)
nm.mean(scores)
knn.fit(Xhh,Yhh)
Ytest = knn.predict(Xtest)
Ytest
fim = pd.DataFrame(Ytest, columns = ["Target"])
fim.to_csv("prediction2.csv", index_label="Id")
fim
