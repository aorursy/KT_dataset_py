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
import sklearn.datasets

newsgroups=sklearn.datasets.fetch_20newsgroups_vectorized()

X,y=newsgroups.data, newsgroups.target

X.shape

y.shape



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X,y)

y_pred=knn.predict(X)

knn.score(X,y)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =  train_test_split(X,y)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)



from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train, y_train) 

lr.predict(X_test) 



import sklearn.datasets

wine=sklearn.datasets.load_wine()

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(wine.data,wine.target)

lr.score(wine.data,wine.target)

lr.predict_proba(wine.data[:1])



x=np.arange(3)

y=np.arange(3,6)

x*y

np.sum(x*y)

x@y



from scipy.optimize import minimize

minimize(np.square,0).x

minimize(np.square,2).x



import sklearn.datasets

wine=sklearn.datasets.load_wine()

from sklearn.svm import LinearSVC

svm=LinearSVC()

svm.fit(wine.data,wine.target)

svm.score(wine.data,wine.target)



import sklearn.datasets

wine=sklearn.datasets.load_wine()

from skelarn.svm import SVC

svm=SVC() # default hyperparameters

svm.fit(wine.data, wine.target);

svm.score(wine.data,wine.target)



from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X,y)

lr.predict(X)[10]

lr.predict(X)[20]

lr.coef_@X[10]+lr.intercept_ #raw model output

lr.coef_@X[20]+lr.intercept_ #raw model output
