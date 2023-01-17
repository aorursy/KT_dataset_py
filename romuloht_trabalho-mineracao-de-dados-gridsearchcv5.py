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
X_train = train.values[:, 1:]

y_train = train.values[:, 0]

X_test = test.values[:, 1:]

y_test = test.values[:, 0]
len(X_train), len(y_train), len(X_test), len(y_test)
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import metrics
mlp = MLPClassifier(activation='logistic', learning_rate_init=0.001)

knn = KNeighborsClassifier(n_neighbors=3)

rfc = RandomForestClassifier(n_estimators=200)

gbc = GradientBoostingClassifier(max_depth=7)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

metrics.accuracy_score(y_test, y_pred_mlp)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

metrics.accuracy_score(y_test, y_pred_knn)
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_rfc)
gbc.fit(X_train, y_train)

y_pred_gbc = gbc.predict(X_test)

metrics.accuracy_score(y_test, y_pred_gbc)