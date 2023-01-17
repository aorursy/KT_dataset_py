import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline



import pandas as pd

import numpy as np



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
def plot_clf(clf, X, y, txt=None, n=100):

    

    try:

        y_pred = clf.predict_proba(X)[:, 1]

    except AttributeError:

        y_pred = clf.predict(X)

    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),

                         np.linspace(-3, 3, 500))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)

    

    plt.figure(figsize=(8, 7))

    plt.scatter(X[:n, 0], X[:n, 1], c=y[:n], s=15, alpha=1.0)

    plt.contourf(xx, yy, z, levels=np.linspace(z.min(), z.max(), 20), cmap=plt.cm.bwr, alpha=0.3)

    plt.xlim([-3, 3])

    plt.ylim([-3, 3])

    plt.axis('off')

    plt.title(f'{txt} {np.round(roc_auc_score(y, y_pred), 3)}')
from sklearn.datasets import make_moons



X, y = make_moons(10000, noise=0.5)



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1999)



plt.figure(figsize=(10,10))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, alpha=0.2, label='train')

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=5, alpha=1.0, label='test')

plt.xlabel("first feature")

plt.ylabel("second feature")

plt.legend()

plt.show()
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge, Lasso, LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBClassifier



lr = LogisticRegression(penalty='none', solver='saga')

lr.fit(X_train, y_train)

plot_clf(lr, X_test, y_test, 'LR', n=300)



rg = Ridge(alpha=0.1)

rg.fit(X_train, y_train)

plot_clf(rg, X_test, y_test, 'Ridge 0.1', n=300)



ls = Lasso(alpha=0.1)

ls.fit(X_train, y_train)

plot_clf(ls, X_test, y_test, 'Lasso 0.1', n=300)



knn3 = KNeighborsRegressor(n_neighbors=3)

knn3.fit(X_train, y_train)

plot_clf(knn3, X_test, y_test, '3NN', n=300)



knn10 = KNeighborsRegressor(n_neighbors=10)

knn10.fit(X_train, y_train)

plot_clf(knn10, X_test, y_test, '10NN', n=300)



dt = RandomForestRegressor(n_estimators=1, max_depth=10)

dt.fit(X_train, y_train)

plot_clf(dt, X_test, y_test, 'DT 10', n=300)



rf1 = RandomForestRegressor(n_estimators=30, max_depth=1)

rf1.fit(X_train, y_train)

plot_clf(rf1, X_test, y_test, 'RF 30;1', n=300)



rf5 = RandomForestRegressor(n_estimators=30, max_depth=10)

rf5.fit(X_train, y_train)

plot_clf(rf5, X_test, y_test, 'RF 30;10', n=300)



xgb = XGBClassifier(n_estimators=30, silent=True)

xgb.fit(X_train, y_train)

plot_clf(xgb, X_test, y_test, 'XGB 30;5', n=300)
