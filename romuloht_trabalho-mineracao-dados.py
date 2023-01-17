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
train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
train.info()
test.info()
train.head()
test.head()
X_train = train.values[:, 1:]

y_train = train.values[:, 0]

X_test = test.values[:, 1:]

y_test = test.values[:, 0]
len(X_train), len(y_train), len(X_test), len(y_test)
y_train
y_test
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics
mlp = MLPClassifier()

knn = KNeighborsClassifier()

svc = SVC()

dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier()

abc = AdaBoostClassifier()

gbc = GradientBoostingClassifier()

gnb = GaussianNB()

qda = QuadraticDiscriminantAnalysis()
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

metrics.accuracy_score(y_test, y_pred_mlp)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

metrics.accuracy_score(y_test, y_pred_knn)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_svc)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_dtc)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_rfc)
abc.fit(X_train, y_train)
y_pred_abc = abc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_abc)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_gbc)
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

metrics.accuracy_score(y_test, y_pred_gnb)
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

metrics.accuracy_score(y_test, y_pred_qda)