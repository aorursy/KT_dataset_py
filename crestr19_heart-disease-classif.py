# This Python 3 environment comes with many helpful analytics libraries installed# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC,SVC

from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_decision_regions

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')

data.head()
f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.bar(data.age.value_counts().index, data.age.value_counts().values)

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
print('The minimum age in the dataset is {}, the maximum age is {} and the mean is {}'.format(np.min(data.age.values), np.max(data.age.values), np.mean(data.age.values)))

f, ax = plt.subplots(1, 1, figsize=(12, 3))

ax.bar(data.sex.value_counts().index, data.sex.value_counts().values)

plt.xlabel('Sex')

plt.ylabel('Count')

plt.show()
print('There are {} men in the dataset and {} women'.format(np.count_nonzero(data.sex.values== 1), np.count_nonzero(data.sex.values==0)))

print('There are {} men with hearth disease (target == 1) and {} men "healthy"'.format(np.count_nonzero(data.sex.values & data.target.values), np.count_nonzero(data.sex.values & 1-data.target.values)))

print('There are {} women with hearth disease (target == 1) and {} women "healthy"'.format(np.count_nonzero(1-data.sex.values & data.target.values), np.count_nonzero(1-data.sex.values & 1-data.target.values)))

np.min(data.values[:,12])
#Let's see some of the people with target == 1

data.head()
#Let's see some of the people with target == 0

data.tail()
dummy_sex = pd.get_dummies(data["sex"],prefix="sex")

dummy_sex.columns.values[0] = "Women"

dummy_sex.columns.values[1] = "Men"

dummy_sex.head()
data = data.drop(["sex"],axis = 1)
data = pd.concat([dummy_sex,data],axis=1)
data.head()
data_cat = data[["Women","Men","age","cp","fbs","restecg","exang","slope","ca","thal","target"]]

data_cat.head()
data_con = data[["trestbps","chol","thalach","oldpeak","target"]]

data_con.head()
X = data.values[:,0:2].astype(float)

Y = data.values[:,14].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)
clasificador = LinearSVC(C=0.0001)

clasificador.fit(X_train, y_train)

print(clasificador.score(X_test, y_test))

plot_decision_regions(X_test, y_test, clf=clasificador, legend=2)
X = data_con.values[:,0:-1].astype(float)

Y = data.values[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)
clasificador = LinearSVC(C=0.0001)

clasificador.fit(X_train, y_train)

print(clasificador.score(X_test, y_test))

X = data_cat.values[:,0:-1].astype(float)

Y = data.values[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)
clasificador = LinearSVC(C=0.0001)

clasificador.fit(X_train, y_train)

print(clasificador.score(X_test, y_test))
X = data.values[:,0:-1].astype(float)

Y = data.values[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)
clasificador = LinearSVC(C=0.0001)

clasificador.fit(X_train, y_train)

print(clasificador.score(X_test, y_test))
X = data_con.values[:,0:-1].astype(float)

Y = data_con.values[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)
std = StandardScaler()

pca = PCA(n_components=5)

clas = LinearSVC(C=200.0, max_iter=10000, tol=0.01)

clas2 = SVC(C=200.0, kernel='rbf')

pipe = Pipeline([('std', std), ('pca', pca), ('clas', clas)])

param_grid = {'pca__n_components': [4, 3, 2],

              'clas__C': [0.1, 1.0, 10.0, 100.0, 1000.0],

              'clas': [clas, clas2]}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
search.fit(X_train, y_train)
search.best_estimator_.score(X_test, y_test)
search.best_params_ # Best parameters for the continuous data
X = data.values[:,0:-1].astype(float)

Y = data.values[:,-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50)
search.fit(X_train, y_train)
search.best_estimator_.score(X_test, y_test)
search.best_params_ # Best parameters for the continuous data
pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]