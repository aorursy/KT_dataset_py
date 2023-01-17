# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegressionCV

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import auc,confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/creditcard.csv')

data=pd.DataFrame(data)
print(data.info())

print('______________________________________\n')

print(data.describe())

print('______________________________________\n')

data.head(15)
dt=data.copy()

dt=dt.loc[:,'Time':'V28']

dt.head()
miss=(len(data)-data.count())*100.0/len(data)

print(miss)

print('\nThe number of frauds:',data.loc[data.Class == 1, 'Class'].count())
X_train,X_test,y_train,y_test=train_test_split(dt,data[['Class']],test_size=0.33\

                                           ,random_state=6)
DTC = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

DTC.fit(X_train, y_train)

y_pred = DTC.predict(X_test)

print(DTC.score(X_test,y_test))

cm = (confusion_matrix(y_test,y_pred))

cm
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(knn.score(X_test,y_test))

cm = (confusion_matrix(y_test,y_pred))

cm
nbc = GaussianNB()

nbc.fit(X_train,y_train)

y_pred = nbc.predict(X_test)

print(nbc.score(X_test,y_test))

cm = (confusion_matrix(y_test,y_pred))

cm
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print(rfc.score(X_test,y_test))

cm = (confusion_matrix(y_test,y_pred))

cm
from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8),n_estimators=600)

ab = RandomForestClassifier()

ab.fit(X_train,y_train)

y_pred = ab.predict(X_test)

print(ab.score(X_test,y_test))

cm = (confusion_matrix(y_test,y_pred))

cm