import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns
adult=pd.read_csv("adult.data",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult = pd.read_csv("adult.test.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
nadult=adult.dropna()

nTestAdult=testAdult.dropna()
adult['Occupation'].describe()
adult['Age'].describe()
adult['Workclass'].describe()
adult['Education'].describe()
adult['Education-Num'].describe()
adult['Martial Status'].describe()
adult['Relationship'].describe()
adult['Race'].describe()
adult['Sex'].describe()
adult['Capital Gain'].describe()
adult["Country"].value_counts().head()
sns.distplot(adult['Age'], bins=70)

plt.ylabel('Quantity')
sns.distplot(adult['Hours per week'], bins=70)

plt.ylabel('Quantity')
plt.figure(figsize=(30,10))

adult['Occupation'].hist(bins=14)

plt.xlabel('Occupation')

plt.ylabel('Quantity')

plt.title('Occupation Histogram')
plt.figure(figsize=(30,10))

adult['Capital Gain'].hist(bins=10)

plt.xlabel('Capital Gain')

plt.ylabel('Quantity')

plt.title('Capital Gain Histogram')
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
plt.figure(figsize=(20, 10))

sns.heatmap(numAdult.corr(),

            annot = True,

            fmt = '.2f',

            cmap='Blues')

plt.title('Correlação entre variáveis do dataset de Adult')

plt.show()
Xadult = numAdult[["Age", "Education-Num",

        "Occupation","Martial Status", "Capital Gain"

        ]]
XtestAdult = numTestAdult[["Age", "Education-Num",

        "Occupation","Martial Status", "Capital Gain"

        ]]

YtestAdult = numTestAdult.Target
YtestAdult = numTestAdult.Target

Yadult = numAdult.Target
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=30)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred=knn.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(YtestAdult,YtestPred)