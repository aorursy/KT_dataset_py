import pandas as pd
import numpy as np
import sklearn
adult = pd.read_csv("../input/adultdataset/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        skiprows=1,
        na_values="?")
adult.drop(adult.index[[0]])
nadult = adult.dropna()
testAdult = pd.read_csv("../input/adultdataset/test_data.csv",
        names=[
        "id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        skiprows=1,
        na_values="?")
testAdult.drop(testAdult.index[[0]])
Xadult = nadult[["Age","Education-Num","Capital Gain","Capital Loss","Hours per week"]]
Yadult = nadult.Target
XtestAdult = testAdult[["Age","Education-Num","Capital Gain","Capital Loss","Hours per week"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=20)
scores
scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
result = np.vstack((testAdult["id"], YtestPred)).T
x = ["id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("Resultados.csv", index = False)
Resultado
