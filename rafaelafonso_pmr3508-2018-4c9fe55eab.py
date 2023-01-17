import pandas as pd
import sklearn
adult = pd.read_csv("../input/baseadult/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
import matplotlib.pyplot as plt
adult["sex"].value_counts()
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="pie")
adult["age"].value_counts().plot(kind="bar")
adult["age"].mean()
nadult = adult.dropna()
nadult.shape
testAdult = pd.read_csv("../input/baseadult/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nTestAdult = testAdult.dropna()
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain","relationship","native.country"]]
Yadult = numAdult.income
XtestAdult = numTestAdult[["age","education.num","capital.gain","relationship","native.country"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=52)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
#from sklearn.metrics import accuracy_score
#accuracy_score(YtestAdult,YtestPred)