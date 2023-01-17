import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
adult = pd.read_csv("../input/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
adult["income"].value_counts()/(adult["income"].value_counts().sum())
adult.isnull().sum().plot(kind="bar")
nadult = adult.dropna()
nadult["native.country"].value_counts()
adult["sex"].value_counts().plot(kind="bar")
adult["marital.status"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
Xadult = (nadult.apply(preprocessing.LabelEncoder().fit_transform)).iloc[:,1:15]
Xcont = Xadult[["age", "fnlwgt", "education.num","capital.gain", "capital.loss", "hours.per.week"]]
Xcontnor = (Xcont - Xcont.min())/(Xcont.max() - Xcont.min())
Xcat = Xadult[["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]]
Yadult = nadult.income
scoresknn = []
indice = []
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, Xcont, Yadult, cv=10)
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, Xcontnor, Yadult, cv=10)
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
mscores = []
for x in range (1, 50):
    knn = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(knn, Xcat, Yadult, cv=10)
    mscores.append(scores.mean())
scoresknn.append(max(mscores))
indice.append(mscores.index(max(mscores)))
scoresknn
indice
scoresnbg = []
nbg = naive_bayes.GaussianNB()
scores = cross_val_score(nbg, Xadult, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, Xcont, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, Xcontnor, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbg, Xcat, Yadult, cv=10)
scoresnbg.append(scores.mean())
scoresnbg
scoresnbg = []
nbm = naive_bayes.MultinomialNB()
scores = cross_val_score(nbm, Xadult, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbm, Xcont, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbm, Xcontnor, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbm, Xcat, Yadult, cv=10)
scoresnbg.append(scores.mean())
scoresnbg
scoresnbg = []
nbc = naive_bayes.ComplementNB()
scores = cross_val_score(nbc, Xadult, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbc, Xcont, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbc, Xcontnor, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbc, Xcat, Yadult, cv=10)
scoresnbg.append(scores.mean())
scoresnbg
scoresnbg = []
nbb = naive_bayes.BernoulliNB()
scores = cross_val_score(nbb, Xadult, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbb, Xcont, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbb, Xcontnor, Yadult, cv=10)
scoresnbg.append(scores.mean())
scores = cross_val_score(nbb, Xcat, Yadult, cv=10)
scoresnbg.append(scores.mean())
scoresnbg
scoreslr = []
lr = LogisticRegression()
scores = cross_val_score(lr, Xadult, Yadult, cv=10)
scoreslr.append(scores.mean())
scores = cross_val_score(lr, Xcont, Yadult, cv=10)
scoreslr.append(scores.mean())
scores = cross_val_score(lr, Xcontnor, Yadult, cv=10)
scoreslr.append(scores.mean())
scores = cross_val_score(lr, Xcat, Yadult, cv=10)
scoreslr.append(scores.mean())
scoreslr
knn = KNeighborsClassifier(n_neighbors=34)
scores = cross_val_score(knn, Xcontnor, Yadult, cv=10)
scores.mean()
test = pd.read_csv("../input/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
ntest = test.dropna()
Xtest = (nadult.apply(preprocessing.LabelEncoder().fit_transform)).iloc[:,1:15]
Xtestcont = Xtest[["age", "fnlwgt", "education.num","capital.gain", "capital.loss", "hours.per.week"]]
Xtestecontnor = (Xtestcont - Xtestcont.min())/(Xtestcont.max() - Xtestcont.min())
knn.fit(Xcontnor,Yadult)
Ytest = knn.predict(Xtestecontnor)
Ytest
Ytabela = pd.DataFrame(columns=['income'])
Ytabela['income'] = Ytest
Ytabela