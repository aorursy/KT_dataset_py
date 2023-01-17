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
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df['quality'].value_counts().sort_values(ascending=False).plot.bar()
df_1 = df.copy()

df_1[['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'alcohol']] = df_1[['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'alcohol']]/100



y = df_1['quality'] 

X = df_1.drop('quality', axis=1)



X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, 

                                                      y, 

                                                      test_size=0.3, 

                                                      random_state=19)

wine_df = pd.DataFrame(X_train, columns=X.columns)

grr = pd.plotting.scatter_matrix(wine_df, 

                                 c=y_train, 

                                 figsize=(15, 15), 

                                 marker='o',

                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
from sklearn.neighbors import KNeighborsRegressor 

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)
# Вычисляем метрику (меру) качества

knn.score(X_valid, y_valid)
from sklearn.neighbors import KNeighborsRegressor



acc_score = []

for i in range(1,100):

    knn = KNeighborsRegressor(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    acc_score.append(knn.score(X_valid, y_valid))

    

(pd.Series(acc_score)).plot()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)

knn.score(X_valid, y_valid)
from sklearn.neighbors import KNeighborsClassifier



acc_score = []

for i in range(1,51):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    acc_score.append(knn.score(X_valid, y_valid))

    

(pd.Series(acc_score)).plot()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)

scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

print(scores)

max_score = scores.max()

print(max_score)
from sklearn.neighbors import KNeighborsClassifier



acc_score = []

for i in range(1,51):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

    acc_score.append(scores.mean())

    

(pd.Series(acc_score)).plot()

from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 50)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=5)

knn_grid.fit(X_train, y_train)

scores = cross_val_score(knn_grid, X, y, 

                         cv=kf, scoring='accuracy')
knn_grid.best_score_

knn_grid.best_params_
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



acc_score = []

for P in range(20,200):

    knn = KNeighborsClassifier(n_neighbors=24, p=P/20, weights='distance')

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    scores = cross_val_score(knn, X, y, 

                         cv=5, scoring='accuracy')

    acc_score.append(scores.mean())



plt.plot(acc_score)
from sklearn.neighbors import RadiusNeighborsClassifier 



acc_score = []

for i in range(1,51):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    knn = RadiusNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, X, y, 

                         cv=kf, scoring='accuracy')

    acc_score.append(scores.mean())

    

(pd.Series(acc_score)).plot()
y = (df_1['quality'] > 6).astype(int)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, 

                                                      y, 

                                                      test_size=0.3, 

                                                      random_state=19)
from sklearn.neighbors import KNeighborsClassifier



acc_score = []

for i in range(1,51):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    acc_score.append(knn.score(X_valid, y_valid))

    

(pd.Series(acc_score)).plot()
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 50)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=5)

knn_grid.fit(X_train, y_train)

scores = cross_val_score(knn_grid, X, y, 

                         cv=kf, scoring='accuracy')
knn_grid.best_score_
knn_grid.best_params_