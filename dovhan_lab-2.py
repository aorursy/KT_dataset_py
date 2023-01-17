# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import RadiusNeighborsRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
d = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv')

d.head()

d1 = d.drop(['url'], axis=1)

d1.head()
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

StandardScaler(d1)
from sklearn.model_selection import train_test_split

X = d1.drop('shares', axis=1) 

y = d1['shares'] 

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)

X.head()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X_train, y_train)
y_pred=tree.predict(X_valid)
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(X, y)
from sklearn.metrics import accuracy_score

print('Качество модели:', accuracy_score(y_valid, y_pred))



from sklearn.metrics import mean_squared_error

print('Качество модели:', mean_squared_error(y_valid, y_pred))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=12) 

tree = DecisionTreeClassifier(max_depth=10)

scores = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))

from sklearn.model_selection import GridSearchCV



tree_params={'max_depth': np.arange(2, 15)} # словарь параметров (ключ: набор возможных значений)

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy')# перекрестная проверка по 5 блокам

tree_grid.fit(X_train, y_train)
print(tree_grid.best_params_)



print(tree_grid.best_estimator_)
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt

results_d = pd.DataFrame(tree_grid.cv_results_)

plt.plot(results_d['param_max_depth'], results_d['mean_test_score'])



# Sign the axes and graph

plt.xlabel('max_depth')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
rnr = RadiusNeighborsRegressor(radius = 1.5)

rnr.fit(X_train, y_train)

y_pred = rnr.predict(X_valid)

y_pred
rnr = RadiusNeighborsRegressor(radius = 20)

rnr.fit(X_train, y_train)

y_pred = rnr.predict(X_valid)

y_pred
rnr.score(X_valid, y_valid)

mean_squared_error(y_valid, y_pred)
