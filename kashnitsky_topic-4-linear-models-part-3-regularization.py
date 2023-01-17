# we don't like warnings

# you can comment the following 2 lines if you'd like to

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.model_selection import GridSearchCV
# loading data

data = pd.read_csv('../input/microchip_tests.txt',

                   header=None, names = ('test1','test2','released'))

# getting some info about dataframe

data.info()
data.head(5)
data.tail(5)
X = data.iloc[:,:2].values

y = data.iloc[:,2].values
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Released')

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', label='Faulty')

plt.xlabel("Test 1")

plt.ylabel("Test 2")

plt.title('2 tests of microchips. Logit with C=1')

plt.legend();
def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):

    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1

    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),

                         np.arange(y_min, y_max, grid_step))





    # to every point from [x_min, m_max]x[y_min, y_max]

    # we put in correspondence its own color

    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))

    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
poly = PolynomialFeatures(degree=7)

X_poly = poly.fit_transform(X)
X_poly.shape
C = 1e-2

logit = LogisticRegression(C=C, random_state=17)

logit.fit(X_poly, y)



plot_boundary(logit, X, y, grid_step=.01, poly_featurizer=poly)



plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Released')

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', label='Faulty')

plt.xlabel("Test 1")

plt.ylabel("Test 2")

plt.title('2 tests of microchips. Logit with C=%s' % C)

plt.legend();



print("Accuracy on training set:", 

      round(logit.score(X_poly, y), 3))
C = 1

logit = LogisticRegression(C=C, random_state=17)

logit.fit(X_poly, y)



plot_boundary(logit, X, y, grid_step=.005, poly_featurizer=poly)



plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Released')

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', label='Faulty')

plt.xlabel("Test 1")

plt.ylabel("Test 2")

plt.title('2 tests of microchips. Logit with C=%s' % C)

plt.legend();



print("Accuracy on training set:", 

      round(logit.score(X_poly, y), 3))
C = 1e4

logit = LogisticRegression(C=C, random_state=17)

logit.fit(X_poly, y)



plot_boundary(logit, X, y, grid_step=.005, poly_featurizer=poly)



plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Released')

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', label='Faulty')

plt.xlabel("Test 1")

plt.ylabel("Test 2")

plt.title('2 tests of microchips. Logit with C=%s' % C)

plt.legend();



print("Accuracy on training set:", 

      round(logit.score(X_poly, y), 3))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)



c_values = np.logspace(-2, 3, 500)



logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1, n_jobs=-1)

logit_searcher.fit(X_poly, y)
logit_searcher.C_
plt.plot(c_values, np.mean(logit_searcher.scores_[1], axis=0))

plt.xlabel('C')

plt.ylabel('Mean CV-accuracy');
plt.plot(c_values, np.mean(logit_searcher.scores_[1], axis=0))

plt.xlabel('C')

plt.ylabel('Mean CV-accuracy');

plt.xlim((0,10));