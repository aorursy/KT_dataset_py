# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_boston



from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR



from itertools import combinations



import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,15)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston["MEDV"] = boston_dataset.target

boston.head()
corr = boston.corr().abs()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
X = boston[['LSTAT']].values

y = boston['MEDV'].values



lr = LinearRegression()



quad = PolynomialFeatures(2)

cub = PolynomialFeatures(3)



X_quad = quad.fit_transform(X)

X_cub = cub.fit_transform(X)
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

lr.fit(X, y)

y_lin_fit = lr.predict(X_fit)

linear_r2 = r2_score(y, lr.predict(X))
lr.fit(X_quad, y)

y_quad_fit = lr.predict(quad.transform(X_fit))

quad_r2 = r2_score(y, lr.predict(X_quad))
lr.fit(X_cub, y)

y_cub_fit = lr.predict(cub.transform(X_fit))

cub_r2 = r2_score(y, lr.predict(X_cub))
dtr = DecisionTreeRegressor(max_leaf_nodes=15)

dtr.fit(X, y)

y_tree_fit = dtr.predict(X_fit)

tree_r2 = r2_score(y, dtr.predict(X))
svr = SVR(C=3)
svr.fit(X, y)

y_svr_fit = svr.predict(X_fit)

svr_r2 = r2_score(y, svr.predict(X))
plt.scatter(X, y, label="training points", color="darkgray", lw=5)

plt.plot(X_fit, y_lin_fit, label="linear d=1 $R^2=%.2f$" % linear_r2,

         color='blue', lw=2, linestyle=":")

plt.plot(X_fit, y_quad_fit, label="quadr d=2 $R^2=%.2f$" % quad_r2,

         color='red', lw=2, linestyle="-")

plt.plot(X_fit, y_cub_fit, label="cubic d=3 $R^2=%.2f$" % cub_r2,

         color='green', lw=2, linestyle="--")

plt.plot(X_fit, y_tree_fit, label="RegTree d=1 $R^2=%.2f$" % tree_r2,

         color='cyan', lw=2, linestyle="--")

plt.plot(X_fit, y_svr_fit, label="SVR d=1 $R^2=%.2f$" % svr_r2,

         color='purple', lw=2, linestyle="-")

plt.xlabel("% lower status of the population [LSTAT]")

plt.ylabel('Price [MEDV]')

plt.legend(loc="upper right")

plt.show()
X = boston.drop("MEDV", axis=1)

y = boston['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clfs = {

    "lr d=1": Pipeline([('scaler', StandardScaler()), ('lr', lr)]),

    "lr d=2": Pipeline([('scaler', StandardScaler()), ("pf", quad), ('lr', lr)]),

    "lr d=3": Pipeline([('scaler', StandardScaler()), ("pf", cub), ('lr', lr)]),

    "dtr": dtr,

    "svr": Pipeline([('scaler', StandardScaler()), ('svr', svr)])

}
scores = {}

for clf_i in clfs:

    clf = clfs[clf_i]

    min_score_mse = 999999999999

    min_score_r2 = None

    best_features = None

    for num in range(1, len(X)+1):

        for featureset in combinations(X.columns, num):

            clf.fit(X_train[list(featureset)], y_train)

            predicted = clf.predict(X_test[list(featureset)])

            mse = mean_squared_error(y_test, predicted)

            r2 = r2_score(y_test, predicted)

            #print(clf_i, featureset, "r2: %s\tmse:%s" % (r2, mse))

            if min_score_mse > mse:

                min_score_mse = mse

                min_score_r2 = r2

                best_features = featureset

    scores[clf_i] = {"featureset": best_features,

                     "mse": min_score_mse,

                     "r2": min_score_r2}
for clf in scores:

    print(clf, scores[clf])