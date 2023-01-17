# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import neighbors

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import accuracy_score, mean_squared_error



from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.describe()
data.dtypes
from scipy.stats import normaltest

normaltest(data['quality'])
data['quality'].hist(bins=15);
plt.figure(figsize = (12,12))

sns.heatmap(data = data.corr(), annot = True, square = True, cbar = True)
sc = StandardScaler()

X2 = data.drop(['quality'], axis = 1)

y = data['quality']

X = sc.fit_transform(X2)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy_score(y_test, y_pred)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

mean_squared_error(y_test, y_pred)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'n_neighbors': np.arange(1, 50)}

score = 'accuracy'

gscv = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, cv=kf, scoring=score)

gscv.fit(x_train, y_train)

for mean, params in zip(gscv.cv_results_['mean_test_score'],gscv.cv_results_['params']):

    print("\t%s = %0.3f  for %r" % (score, mean, params))
results_data = pd.DataFrame(gscv.cv_results_)



plt.plot(results_data['param_n_neighbors'], results_data['mean_test_score'])



plt.xlabel('k')

plt.ylabel('Тчность')

plt.show()
p_weights = {"p": np.linspace(1,10,200)}



knn_weights = KNeighborsClassifier(n_neighbors=1, weights = "distance")

knn_weights.fit(x_train,y_train)



knn_weights_cv = model_selection.GridSearchCV(knn_weights, p_weights, scoring="accuracy", cv = kf)

knn_weights_cv.fit(x_train, y_train)
knn_weights_cv.best_estimator_
knn_weights_cv.best_score_
from sklearn.neighbors import RadiusNeighborsClassifier



r_neigh_classifier = RadiusNeighborsClassifier(radius = 5)

r_neigh_classifier.fit(X,y)

y_pred = r_neigh_classifier.predict(x_test)

print(r_neigh_classifier.score(x_test, y_test))
from sklearn.neighbors import RadiusNeighborsRegressor



r_neigh_regressor = RadiusNeighborsRegressor(radius = 5)

r_neigh_regressor.fit(X,y)

y_pred = r_neigh_regressor.predict(x_test)

print(r_neigh_regressor.score(x_test, y_test))
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()

nc.fit(x_train, y_train)

y3_pred = nc.predict(x_test)

nc.score(x_test, y_test)