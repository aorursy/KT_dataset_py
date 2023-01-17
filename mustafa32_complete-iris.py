# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/iris/Iris.csv")

df.head()
sns.heatmap(df.corr())
sns.scatterplot(df['SepalLengthCm'],df['SepalWidthCm'])
sns.scatterplot(df['PetalLengthCm'],df['PetalWidthCm'])
x = df.iloc[:,1:5].values

y = df.Species.values
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logr = LogisticRegression(random_state=0)

logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.svm import SVC

svc = SVC(kernel='poly')

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print(accuracy_score(y_pred,y_test))