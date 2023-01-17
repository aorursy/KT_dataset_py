import pandas as pd 
import sklearn
household = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
household
Xhousehold = household[["SQBage","SQBescolari","computer","television"]]
Yhousehold = household.Target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xhousehold, Yhousehold, cv=10)
scores
knn.fit(Xhousehold,Yhousehold)
testhousehold = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testhousehold
Xtesthousehold = testhousehold [["SQBage","SQBescolari","computer","television"]]
YtestPred = knn.predict(Xtesthousehold)
YtestPred
arr1= testhousehold.iloc[:,0].values
arr1 = arr1.ravel()
dataset = pd.DataFrame({'Id':arr1[:],'Target':YtestPred[:]})
dataset.to_csv("HHIcompetition.csv", index = False)
