# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

df.head().T
df['income'].hist(bins=5)
df1 = df.drop(['education','fnlwgt','native-country','relationship'], axis = 1)

print(df1.workclass.mode())

print(df1.occupation.mode())

df1['workclass'].replace({"?":"Other"}, inplace = True)

df1['occupation'].replace({"?":"Prof-specialty"}, inplace = True)

df1['income'] = df1['income'].map({'<=50K' : 0, '>50K' : 1})

df1['gender'] = df1['gender'].map({'Male' : 0, 'Female' : 1})

df1['race'] = df1['race'].map({'White' : 0, 'Black' : 1, 'Other' : 2, 'Amer-Indian-Eskimo': 3, 'Asian-Pac-Islander': 4})

df1 = pd.get_dummies(df1)

df1.head().T
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df2 = df1.drop('income', axis=1) 

X = scaler.fit_transform(df2)
from sklearn.model_selection import train_test_split



y = df1['income'] 



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=11)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, n_jobs = -1)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_valid)

y_pred

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

print("Accuracy score: ", accuracy_score(y_valid, y_pred), "\n")

print(classification_report(y_valid,y_pred))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=7)

scores = cross_val_score(knn, X, y, cv=kf, scoring='f1')

print(scores)

avg = scores.mean()

print(avg)

from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51, 2)}

knn_grid = GridSearchCV(knn, knn_params, scoring='f1',cv=kf, verbose = 159)

knn_grid.fit(X_valid, y_valid)

knn_grid.best_estimator_
import matplotlib.pyplot as plt

res = pd.DataFrame(knn_grid.cv_results_)

res.head().T

plt.plot(res['param_n_neighbors'], res['mean_test_score'])

plt.ylabel('accuracy')

plt.xlabel('number of neighbors')

plt.show()
knn = KNeighborsClassifier(metric='minkowski', n_neighbors=9,

           weights='distance', n_jobs = -1)



knn_params = {'p': np.linspace(1.0, 10.0,num=100)}

knn_grid = GridSearchCV(knn, knn_params, scoring='f1',cv=kf, verbose = 100)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=11)

knn_grid.fit(X_valid, y_valid)

knn_grid.best_estimator_

knn_grid.best_score_
res = pd.DataFrame(knn_grid.cv_results_)

res.head().T

plt.plot(res['param_p'], res['mean_test_score'])

plt.ylabel('accuracy')

plt.xlabel('p')

plt.show()
from sklearn.neighbors import RadiusNeighborsClassifier

neigh = RadiusNeighborsClassifier(radius=15, n_jobs = -1)

neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_valid)

print("Accuracy score: ", accuracy_score(y_valid, y_pred), "\n")