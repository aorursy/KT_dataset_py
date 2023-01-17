import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adultdb/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
adult["income"].value_counts()/(adult["income"].value_counts().sum())
adult["native.country"].value_counts()
adult["sex"].value_counts().plot(kind="bar")
adult["marital.status"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
adult.isnull().sum().plot(kind="bar")
nadult = adult.dropna()
nadult
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult.income
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
scores.mean()
knn = KNeighborsClassifier(n_neighbors=26)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
test = pd.read_csv("../input/adultdb/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
ntest = test.dropna()
Xtest = ntest[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
knn.fit(Xadult,Yadult)
Ytest = knn.predict(Xtest)
Ytest
Ytabela = pd.DataFrame(index=ntest.Id,columns=['income'])
Ytabela['income'] = Ytest
Ytabela