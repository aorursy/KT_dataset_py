# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")

df.head()
df.describe()
df.info()
x = df.iloc[:,1:5]

print(x.sample(5))
y = df.iloc[:,5:6]

print(y.sample(5))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0, test_size=0.33)
sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)
modelLR = LogisticRegression(random_state = 0)

modelLR.fit(X_train,y_train)

y_pred = logr.predict(X_test) #tahmin

cm = confusion_matrix(y_pred, y_test)

print(cm)
knn = KNeighborsClassifier(n_neighbors = 1, metric="minkowski")

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_pred,y_test)

print(cm)
modelSVC = SVC(kernel = 'poly')

modelSVC.fit(X_train, y_train)

y_pred = modelSVC.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_pred,y_test)

print(cm)
rfc = RandomForestClassifier(n_estimators=10, criterion = "entropy")

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
y_proba = rfc.predict_proba(X_test)

print(y_test)

print(y_proba[:,0])
fpr, tpr, thold = metrics.roc_curve(y_test,y_proba[:,0], pos_label = 'e')
fpr