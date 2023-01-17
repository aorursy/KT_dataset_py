import pandas as pd 
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult.shape 
adult.head() 
adult["native.country"].value_counts() # numero de dados para cada país
import matplotlib.pyplot as plt
adult.describe()
adult["age"].value_counts().plot(kind="bar")
adult["workclass"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["education.num"].value_counts().plot(kind="bar")
adult["marital.status"].value_counts().plot(kind="bar")
adult["race"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["native.country"].value_counts().plot(kind="bar")
nadult = adult.dropna() # excluindo linhas sem dados
nadult
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult.income
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
scores = 0.0
for k in range(10, 40):

    knn = KNeighborsClassifier(k)

    score = np.mean(cross_val_score(knn, Xadult, Yadult, cv = 10)) 

    

    if score > scores:

        bestK = k

        scores = score

        

print(bestK)
knn = KNeighborsClassifier(n_neighbors=bestK)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
XtestAdult = testAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
XtestAdult
YtestPred = knn.predict(XtestAdult)
YtestPred
YtestPred.shape
prev = pd.DataFrame(columns = ["Id","income"])

prev.Id = XtestAdult.index

prev.income= YtestPred

prev
prev.to_csv("submission.csv", index=False)