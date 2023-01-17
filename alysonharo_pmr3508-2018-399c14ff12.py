import pandas as pd
import numpy as np
import sklearn
adult = pd.read_csv("../input/adultdb/train_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
adult["Country"].value_counts()
import matplotlib.pyplot as plt
adult["Age"].value_counts().plot(kind="bar")

adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult
testAdult = pd.read_csv("../input/adultdb/test_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nTestAdult=testAdult
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
XtestAdult
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
YtestPred = knn.predict(XtestAdult)
YtestPred
result = np.vstack((nTestAdult["Id"], YtestPred)).T
x = ["id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado




