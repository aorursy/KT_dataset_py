# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

df.head()
df.info()
sns.countplot(df['churn'])
sns.countplot(df['state'])
df = df.drop(['state', 'phone number'], axis='columns')
sns.countplot(df["area code"])
df['area code'] = df['area code'].map({408: 0, 415: 1, 510: 2})

df['voice mail plan'] = df['voice mail plan'].map( {"no": 0,"yes": 1} )

df['international plan'] = df['international plan'].map( {"no": 0,"yes": 1} )

df['churn'] = df['churn'].map( {False: 0, True: 1 })

df.head()
scaler = StandardScaler()

X = scaler.fit_transform(df.drop('churn', axis='columns'))

Y = df['churn']

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.25, random_state=42)
kc = KNeighborsClassifier(n_neighbors=1)

kc.fit(X_train, y_train)

Y_res = kc.predict(X_valid)

Y_res
accuracy_score(y_valid, Y_res)
fold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(kc, X, Y,scoring='accuracy', cv=fold)

print(scores, scores.mean())
grid = GridSearchCV(kc, {'n_neighbors': np.arange(1, 15)}, scoring='accuracy', cv=fold)

grid.fit(X_train, y_train)

print(grid.best_params_)

print(grid.best_score_)

grid_res = pd.DataFrame(grid.cv_results_)

grid_res.head()
grid_res.plot(x='param_n_neighbors', y='mean_test_score')
scores = cross_val_score(kc, X, Y, scoring='f1', cv=fold)

print(scores, scores.mean())
kc_w = KNeighborsClassifier(n_neighbors=5, weights = 'distance')

kc_w.fit(X_train, y_train)
kc_w_grid = GridSearchCV(kc_w, {"p": np.linspace(1,10, 20)}, scoring="accuracy", cv = fold)

kc_w_grid.fit(X_train, y_train)
print(kc_w_grid.best_params_)

print(kc_w_grid.best_score_)

grid_res = pd.DataFrame(kc_w_grid.cv_results_)

grid_res.head()
grid_res.plot(x='param_p', y='mean_test_score')
from sklearn.neighbors import RadiusNeighborsClassifier



cf_rad = RadiusNeighborsClassifier(radius=10)

cf_rad.fit(X_train, y_train)



scores = cross_val_score(cf_rad, X, Y, scoring='accuracy', cv=fold)

print(scores, scores.mean())