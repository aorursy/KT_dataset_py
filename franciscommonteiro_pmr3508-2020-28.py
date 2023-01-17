import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import time



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

%matplotlib inline
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="NaN",index_col=0)

testadult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="NaN", index_col=0)
adult.head()
testadult
adult["native.country"].value_counts()
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
nadult = adult.dropna()

ntestadult = testadult.dropna()
nadult
ntestadult
encodenadult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

encodentestadult = ntestadult.apply(preprocessing.LabelEncoder().fit_transform)
encodenadult
encodentestadult
sns.heatmap(encodenadult.corr(), square=True,vmin=-1, vmax=1)

plt.show()
XAdult = encodenadult[["age", "education.num","marital.status", 

        "occupation", "race", "sex","capital.gain", "capital.loss",

        "hours.per.week"]]
YAdult = encodenadult.income
k = 25

kinicial = 25

kfinal = 30

ncv = 5

knn = KNeighborsClassifier(n_neighbors= k , p=2)

scores = cross_val_score(knn, XAdult, YAdult, cv= ncv)

a = [scores.mean()]

while k < kfinal :

    k = k+1

    knn = KNeighborsClassifier(n_neighbors= k , p=2)

    scores = cross_val_score(knn, XAdult, YAdult, cv= ncv)

    a.append(scores.mean())

k = kinicial + a.index(max(a))

knn = KNeighborsClassifier(n_neighbors= k , p=2)

scores = cross_val_score(knn, XAdult, YAdult, cv= ncv)

print ("O numero de validações cruzadas feitas foi ",ncv)

print ("O k usado no KNN foi ",k)

print ("Os Resultados de acuracia da validação cruzada foram " ,scores)

print ("E a Média dos resultados foi" ,scores.mean())