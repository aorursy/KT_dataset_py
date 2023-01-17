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
dataset = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
from sklearn.model_selection import StratifiedKFold
X = dataset.values[:, 1:]

y = dataset.values[:, 0]
len(X), len(y)
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import metrics
mlp = MLPClassifier()

knn = KNeighborsClassifier()

rfc = RandomForestClassifier()

gbc = GradientBoostingClassifier()
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
k = 0

accs = []

for idx_train, idx_val in kfold.split(X, y):

    print("Fold", k)

    k += 1

    mlp.fit(X[idx_train], y[idx_train])

    y_pred = mlp.predict(X[idx_val])

    acc = metrics.accuracy_score(y[idx_val], y_pred)

    accs.append(acc)

    print("Accuracy:", acc)

    print()
np.mean(accs)
k = 0

accs = []

for idx_train, idx_val in kfold.split(X, y):

    print("Fold", k)

    k += 1

    knn.fit(X[idx_train], y[idx_train])

    y_pred = knn.predict(X[idx_val])

    acc = metrics.accuracy_score(y[idx_val], y_pred)

    accs.append(acc)

    print("Accuracy:", acc)

    print()
np.mean(accs)
k = 0

accs = []

for idx_train, idx_val in kfold.split(X, y):

    print("Fold", k)

    k += 1

    rfc.fit(X[idx_train], y[idx_train])

    y_pred = rfc.predict(X[idx_val])

    acc = metrics.accuracy_score(y[idx_val], y_pred)

    accs.append(acc)

    print("Accuracy:", acc)

    print()
np.mean(accs)
k = 0

accs = []

for idx_train, idx_val in kfold.split(X, y):

    print("Fold", k)

    k += 1

    gbc.fit(X[idx_train], y[idx_train])

    y_pred = gbc.predict(X[idx_val])

    acc = metrics.accuracy_score(y[idx_val], y_pred)

    accs.append(acc)

    print("Accuracy:", acc)

    print()
np.mean(accs)