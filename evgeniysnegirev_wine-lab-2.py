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
from sklearn.datasets import load_wine

import matplotlib.pyplot as plt

%matplotlib inline
wine_data = load_wine()

data = pd.DataFrame(wine_data["data"], columns = wine_data["feature_names"])

data["target"] = wine_data["target"]

data.head()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier (random_state = 42)

x = data.drop("target", axis = 1)

y = data["target"]

tree.fit(x,y)

tree.score(x,y)
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

parameters = {

    'max_depth': np.arange(2, 9),

    'max_features': np.linspace(0.3, 1, 8),

    'min_samples_leaf': np.arange(5, 50, 5)

}

x_scaled = scaler.fit_transform(x)

tree = DecisionTreeClassifier (random_state = 42)

kf = KFold(random_state = 42, n_splits = 5, shuffle = True)

cv = GridSearchCV(tree, param_grid = parameters, cv = kf, iid = True)

cv.fit(x_scaled, y)

#clf = cross_val_score (tree, x_scaled, y, cv = 5)

#print(clf.mean())

print(cv.best_score_) 

print(cv.best_params_)
from sklearn.neighbors import KNeighborsClassifier
score = [] 

for i in range (1, 50):

    knn = KNeighborsClassifier(n_neighbors = i) 

    kf = KFold(random_state = 42, n_splits = 5, shuffle = True) 

    cv = cross_val_score(knn, x_scaled, y, cv = kf) 

    score.append(cv.mean())

plt.plot(score)

print(max(score)) 

print(score.index(max(score))+1)