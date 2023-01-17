# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)});

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/adult-income-dataset/adult.csv')

df.head()

X = df.drop('income', axis=1)

y = df['income'].map({'<=50K':0, '>50K':1})

num_features = ['age', 'fnlwgt', 'educational-num', 

                'capital-gain', 'capital-loss', 'hours-per-week']

X_num = X[num_features]

X_cat = X.drop(num_features, axis=1)

print(X.shape, X_num.shape, X_cat.shape)
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

X_cat_new = oe.fit_transform(X_cat)
X_new = np.hstack([X_num.values, X_cat_new])

X_new.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_new)

X_scaled = scaler.transform(X_new)

print(X_scaled)
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, 

                                                   y, 

                                                    test_size=0.25, 

                                                    random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train1, y_train1)
y_pred1 = knn.predict(X_test1)
from sklearn.metrics import accuracy_score

accuracy_score(y_test1, y_pred1)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier()

scores = cross_val_score(knn, X_scaled, y, 

                         cv=kf, scoring='accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51, 10)}  

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=5)  

knn_grid.fit(X_train1, y_train1)
print(knn_grid.scorer_)
print(knn_grid.best_score_)
knn_grid.best_params_
pd.DataFrame(knn_grid.cv_results_).T
y_pred = knn_grid.predict(X_test1)

accuracy_score(y_test1, y_pred)
best_knn = KNeighborsClassifier(n_neighbors= 31)

y_pred = best_knn.fit(X_train1, y_train1).predict(X_test1)

accuracy_score(y_test1, y_pred)
k_metr = []

k = []

for i in range(1,52):

    best_knn = KNeighborsClassifier(n_neighbors= i)

    y_pred = best_knn.fit(X_train1, y_train1).predict(X_test1)

    k_metr.append(accuracy_score(y_test1, y_pred))

    k.append(i)
plt.plot(k_metr, k)

plt.show()
from sklearn.neighbors import RadiusNeighborsClassifier

neigh = RadiusNeighborsClassifier()

neigh.fit(X_scaled, y) 

neigh.score(X_scaled, y)