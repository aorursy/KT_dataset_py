import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
Household = pd.read_csv("../input/train.csv",
        na_values="?")
Household.shape
Household.head()
Household.describe()
Householdtest = pd.read_csv("../input/test.csv",
        na_values='.')
Householdtest.head()
nHousehold = Household.dropna()
nHousehold.head()
Xhouse = Household[["r4m1", "r4t1", "paredblolad", "paredmad", "tipovivi2", "tipovivi3", "qmobilephone", 
                        "lugar4", "SQBedjefe", "SQBhogar_nin", "SQBdependency"]]
Yhouse = Household.Target
Xhousetest = Householdtest[["r4m1", "r4t1", "paredblolad", "paredmad", "tipovivi2", "tipovivi3", "qmobilephone", 
                        "lugar4", "SQBedjefe", "SQBhogar_nin", "SQBdependency"]]
media = np.zeros(70)
for k in range(1,71):
    classficador = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(classficador, Xhouse, Yhouse, cv=10)
    media[k-1] = np.mean(score)
    
np.amax(media)
media
K = 68
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(Xhouse,Yhouse)
YtestHousePred = knn.predict(Xhousetest)
Send = pd.DataFrame(Householdtest.Id)
Send["Id"] = Householdtest.Id
Send["Target"] = YtestHousePred
Send.to_csv("HHprediction.csv", index=False)
