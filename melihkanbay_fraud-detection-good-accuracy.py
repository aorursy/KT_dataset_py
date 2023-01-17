# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
data.info()
data.Class.value_counts()
data.groupby(data.Class).Amount.mean().plot(kind = "bar")

plt.show()

print(data.groupby(data.Class).Amount.mean())
data[data.Amount > data.Amount.mean()].Class.value_counts()
data.Amount.mean()
data.Amount = (data.Amount-data.Amount.min())/(data.Amount.max()-data.Amount.min())
data["amount_mean_up"] = data.Amount

data.amount_mean_up = [1 if i >  88.34961925087359 else 0 for i in data.amount_mean_up]

data.head()
data.groupby(data.Class).Time.mean().plot(kind = "bar")

plt.show()

print(data.groupby(data.Class).Time.mean())
data.Time.mean()
data["time_mean_up"] = data.Time

data.time_mean_up = [1 if i <  94813.85957508067 else 0 for i in data.time_mean_up]

data.head()
data.Time = (data.Time-data.Time.min())/(data.Time.max()-data.Time.min())
data.groupby(data.Class).V4.mean().plot(kind = "bar")

plt.show()

print(data.groupby(data.Class).V4.mean())
data.V4.mean()
data["v4_mean_up"] = data.Time

data.v4_mean_up = [1 if i >  2.782312291808533e-15 else 0 for i in data.v4_mean_up]

data.head()
data.V4 = (data.V4-data.V4.min())/(data.V4.max()-data.V4.min())
y = data.Class.values

X = data.drop("Class",axis = 1).values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score
y_pred = logreg.predict(X_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()
data = data.sample(frac=1,replace = False)

fraud_df = data.loc[data['Class'] == 1]

non_fraud_df = data.loc[data['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



df = normal_distributed_df.sample(frac=1,replace = False, random_state= 42 )
sns.countplot("Class",data = df)

plt.show()
yu = df.Class.values

Xu = df.drop("Class",axis = 1).values

Xu_train, Xu_test, yu_train, yu_test = train_test_split(Xu, yu, test_size=0.33, random_state=42)
logreg = LogisticRegression()

logreg.fit(Xu_train, yu_train)

yu_pred = logreg.predict(Xu_test)

print(metrics.accuracy_score(yu_test, yu_pred))
yu_pred = logreg.predict(Xu_test)

yu_true = yu_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(yu_true,yu_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("yu_pred")

plt.ylabel("yu_true")

plt.title("Confision Matrix")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(Xu_train,yu_train)

yu_pred = knn.predict(Xu_test)

print(metrics.accuracy_score(yu_test, yu_pred))
yu_pred = knn.predict(Xu_test)

yu_true = yu_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(yu_true,yu_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("yu_pred")

plt.ylabel("yu_true")

plt.title("Confision Matrix")

plt.show()
k_range = list(range(1, 31))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, Xu, yu, cv=10, scoring='accuracy')

    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')

plt.show()
from sklearn.model_selection import GridSearchCV

param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

grid.fit(Xu, yu)

print(grid.best_params_,grid.best_score_)
from sklearn.ensemble import RandomForestClassifier

rt=RandomForestClassifier(n_estimators=32,random_state=1)

rt.fit(Xu_train,yu_train)

yu_pred = rt.predict(Xu_test)

print(metrics.accuracy_score(yu_test, yu_pred))
score_list2=[]

for i in range(30,36):

    rt2=RandomForestClassifier(n_estimators=i,random_state=1)

    rt2.fit(Xu_train,yu_train)

    score_list2.append(rt2.score(Xu_test,yu_test))



plt.figure(figsize=(12,8))

plt.plot(range(30,36),score_list2)

plt.xlabel("Esimator values")

plt.ylabel("Acuuracy")

plt.show()
yu_pred = rt.predict(Xu_test)

yu_true = yu_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(yu_true,yu_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("yu_pred")

plt.ylabel("yu_true")

plt.title("Confision Matrix")

plt.show()
rt2=RandomForestClassifier(n_estimators=32,random_state=42)

rt2.fit(Xu_train,yu_train)

y_pred = rt2.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

y_true = y_test


cm=confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Confision Matrix")

plt.show()

plt.show()

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred)*100)