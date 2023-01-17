# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/glass/glass.csv")

data.head()
data.shape
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(data=data,x="Type")
correlation=data.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
data.columns
subset1=data[["Mg","Na","Ba","Type"]]

x=subset1.iloc[:,0:3]

y=subset1.iloc[:,3:]
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import cross_val_score

dummy=DummyClassifier(strategy="most_frequent")

cross_val_score(dummy, x, y, cv=3)
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

cross_val_score(dtree, x, y, cv=3)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

cross_val_score(knn, x, y, cv=3)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=124)

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

ypred=knn.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from sklearn.preprocessing import MinMaxScaler

mmscaler=MinMaxScaler()

X_train=mmscaler.fit_transform(x_train)

X_test=mmscaler.transform(x_test)
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

svmonevsrest = OneVsRestClassifier(SVC(kernel="linear")).fit(X_train, y_train)

ypred=svmonevsrest.predict(X_test)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
x=data.iloc[:,0:9]

y=data.iloc[:,9:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=124)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

selekBest=SelectKBest(chi2, k=5)

x_new_train = selekBest.fit_transform(x_train, y_train)

x_new_test=selekBest.transform(x_test)
knn1=KNeighborsClassifier(n_neighbors=1)

knn1.fit(x_new_train,y_train)

ypred=knn1.predict(x_new_test)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
svmonevsrest = OneVsRestClassifier(SVC(kernel="linear")).fit(x_new_train, y_train)

ypred=svmonevsrest.predict(x_new_test)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
knn1=KNeighborsClassifier(n_neighbors=5)

knn1.fit(x_new_train,y_train)

ypred=knn1.predict(x_new_test)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
dtree=DecisionTreeClassifier()

dtree.fit(x_new_train,y_train)

ypred=dtree.predict(x_new_test)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=1,max_time_mins=80)

tpot.fit(x_new_train, y_train)

ypred=tpot.predict(x_new_test)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))