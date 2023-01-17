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
path_to_data = os.path.join(dirname, filename)

data = pd.read_csv(path_to_data)

labels = data[data.columns[-1]].values

feature_matrix = data[data.columns[:-1]].values
from sklearn.model_selection import train_test_split 

train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(

    feature_matrix, labels, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
# создание модели с указанием гиперпараметра C

clf = LogisticRegression(C=1)

# обучение модели

clf.fit(train_feature_matrix, train_labels)

# предсказание на тестовой выборке

y_pred = clf.predict(test_feature_matrix)
from sklearn.metrics import accuracy_score



accuracy_score(test_labels, y_pred)
from sklearn.model_selection import GridSearchCV

# заново создадим модель, указав солвер

clf = LogisticRegression(solver='saga')



# опишем сетку, по которой будем искать

param_grid = {

    'C': np.arange(1, 5), # также можно указать обычный массив, [1, 2, 3, 4]

    'penalty': ['l1', 'l2'],

}



# создадим объект GridSearchCV

search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')



# запустим поиск

search.fit(feature_matrix, labels)



# выведем наилучшие параметры

print(search.best_params_)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
clf = KNeighborsClassifier()

clf.fit(train_feature_matrix,train_labels)

y_pred = clf.predict(test_feature_matrix)

accuracy_score(test_labels, y_pred)
from sklearn.model_selection import GridSearchCV
params = {

    'n_neighbors' : np.arange(1,10),

    'metric' : ['manhattan', 'euclidean'],

    'weights' : ['uniform', 'distance'],

}

clf_grid = GridSearchCV(clf, params,refit=True, scoring='accuracy')

clf_grid.fit(feature_matrix, labels)

print(clf_grid.best_params_)
clf_optimal = KNeighborsClassifier(n_neighbors = 4, metric = 'manhattan', weights = 'distance')

clf_optimal.fit(train_feature_matrix,train_labels)

optimal_pred = clf_optimal.predict(test_feature_matrix)

accuracy_score(test_labels, optimal_pred)

clf_prob = clf_optimal.predict_proba(test_feature_matrix)
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
unique, freq = np.unique(test_labels, return_counts=True)

freq = list(map(lambda x: x / len(test_labels),freq))



pred_freq = clf_prob.mean(axis=0)

plt.figure(figsize=(10, 8))

plt.bar(3, pred_freq, width=0.4, align="edge", label='prediction')

plt.bar(3, freq, width=-0.4, align="edge", label='real')

plt.ylim(0, 0.54)

plt.legend()

plt.show()

print(pred_freq[2])