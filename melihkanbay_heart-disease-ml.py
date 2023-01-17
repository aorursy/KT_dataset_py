# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True) #do not miss this line

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
data.isnull().sum()
data.target.value_counts().plot.pie(figsize = (6,6),autopct='%.1f')

plt.show()

print(data.target.value_counts())
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True,  fmt= '.1f',ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(x="age",y="thalach",hue="target",data=data,alpha=0.8)

plt.show()
data.loc[:,'N1']=0

data.loc[(data['thalach']>160) & (data['age']<=45),'N1']=1

data.groupby("N1").target.value_counts()
f,ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(x="age",y="chol",hue="target",data=data,alpha=0.8)

plt.show()
data.loc[:,'N2']=0

data.loc[(data['chol']<280) & (data['age']<45),'N2']=1

data.groupby("N2").target.value_counts()
f,ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(x="age",y="oldpeak",hue="target",data=data,alpha=0.8)

plt.show()
data.loc[:,'N3']=1

data.loc[(data['oldpeak']>1.9),'N3']=0

data.groupby("N3").target.value_counts()
data.age = (data.age-data.age.min())/(data.age.max()-data.age.min())

data.trestbps = (data.trestbps-data.trestbps.min())/(data.trestbps.max()-data.trestbps.min())

data.chol = (data.chol-data.chol.min())/(data.chol.max()-data.chol.min())

data.thalach = (data.thalach-data.thalach.min())/(data.thalach.max()-data.thalach.min())

data.oldpeak = (data.oldpeak-data.oldpeak.min())/(data.oldpeak.max()-data.oldpeak.min())
data = pd.get_dummies(data,columns = ["cp"])

data = pd.get_dummies(data,columns = ["restecg"])

data = pd.get_dummies(data,columns = ["ca"])

data = pd.get_dummies(data,columns = ["thal"])

data = pd.get_dummies(data,columns = ["slope"])
data.head()
y = data.target.values

X = data.drop("target",axis = 1).values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn import metrics

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

y_true = y_test

print(metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()
k_range = list(range(1, 26))

scores = []

for i in k_range:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)

plt.show()
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

print(scores.mean())
from sklearn.ensemble import RandomForestClassifier

rt2=RandomForestClassifier(n_estimators=17,random_state=42)

rt2.fit(X_train,y_train)

y_pred = rt2.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

y_true = y_test

n_range = list(range(10, 25))

nscores = []

for i in n_range:

    rt2=RandomForestClassifier(n_estimators=i,random_state=42)

    rt2.fit(X_train,y_train)

    y_pred = rt2.predict(X_test)

    nscores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(n_range, nscores)

plt.show()

cm=confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()