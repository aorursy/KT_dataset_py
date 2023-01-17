import warnings

warnings.filterwarnings("ignore")

import itertools

import numpy as np

import pandas as pd

from sklearn.datasets import make_circles

from mlxtend.plotting import plot_decision_regions

import seaborn.apionly as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib notebook

sns.set()
rng = np.random.RandomState(0)

iris = pd.read_csv("../input/iris/Iris.csv", index_col=["Id"])

X_circle, y_cicle = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

X_regres = 100 * rng.rand(100, 1) + 10

X_xor = rng.randn(300, 2)
iris
sns.pairplot(iris, hue="Species")
data = {

    "regresión": {

        "X": X_regres,

        "y": 200 + 1500 * X_regres[:, 0] + rng.rand(X_regres.shape[0]) * 50000

    },

    "regresión2": {

        "X": X_regres,

        "y": 200 + X_regres[:, 0] ** 4 + rng.rand(X_regres.shape[0]) * 50000000

    },

    "iris": {

        "X": iris.drop("Species", axis=1).values[:, [1, 2]],

        "y": iris.Species.astype("category").cat.codes.values

    },

    "xor": {

        "X": X_xor,

        "y": np.array(np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0), dtype=int)

    },

    "circulo": {

        "X": X_circle,

        "y": y_cicle

    }   



}
from sklearn.linear_model import LinearRegression

plt.rcParams['figure.figsize'] = (15, 12)



tipo = "regresión"

model = LinearRegression()

model.fit(data[tipo]["X"], data[tipo]["y"])

plt.scatter(data[tipo]["X"], data[tipo]["y"])

plt.plot(np.linspace(0, 120), model.predict(np.linspace(0, 120)[:, None]))

plt.show()
tipo = "regresión2"

model = LinearRegression()

model.fit(data[tipo]["X"], data[tipo]["y"])

plt.scatter(data[tipo]["X"], data[tipo]["y"])

plt.plot(np.linspace(0, 120), model.predict(np.linspace(0, 120)[:, None]))

plt.show()
def getOLSCoef(X, y):

    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    alpha = 0

    beta = 0

    return alpha, beta



getOLSCoef(data["regresión"]["X"], data["regresión"]["y"])
from sklearn.linear_model import LogisticRegression



gs = gridspec.GridSpec(2, 2)



for tipo, grd  in zip(["iris", "xor", "circulo"], itertools.product([0, 1], repeat=2)):

    clf = LogisticRegression()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    plt.title(tipo)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor



gs = gridspec.GridSpec(2, 3)



for (tipo, clfClass), grd  in zip([("regresión", KNeighborsRegressor),

                                   ("regresión2", KNeighborsRegressor),

                                   ("iris", KNeighborsClassifier),

                                   ("xor", KNeighborsClassifier),

                                   ("circulo", KNeighborsClassifier)], itertools.product([0, 1, 2], repeat=2)):

    clf = clfClass()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    try:

        fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    except:

        plt.scatter(data[tipo]["X"], data[tipo]["y"])

        plt.plot(np.linspace(0, 120), clf.predict(np.linspace(0, 120)[:, None]))

    plt.title(tipo)
from sklearn.naive_bayes import GaussianNB



gs = gridspec.GridSpec(2, 2)



for tipo, grd  in zip(["iris", "xor", "circulo"], itertools.product([0, 1], repeat=2)):

    clf = GaussianNB()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    plt.title(tipo)
from sklearn.svm import LinearSVC, LinearSVR



gs = gridspec.GridSpec(2, 3)



for (tipo, clfClass), grd  in zip([("regresión", LinearSVR),

                                   ("regresión2", LinearSVR),

                                   ("iris", LinearSVC),

                                   ("xor", LinearSVC),

                                   ("circulo", LinearSVC)], itertools.product([0, 1, 2], repeat=2)):

    clf = clfClass()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    try:

        fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    except:

        plt.scatter(data[tipo]["X"], data[tipo]["y"])

        plt.plot(np.linspace(0, 120), clf.predict(np.linspace(0, 120)[:, None]))

    plt.title(tipo)
from sklearn.svm import SVC, SVR



gs = gridspec.GridSpec(2, 3)



for (tipo, clfClass, params), grd  in zip([("regresión", SVR, {"kernel": "linear"}),

                                   ("regresión2", SVR, {"kernel": "poly"}),

                                   ("iris", SVC, {}),

                                   ("xor", SVC, {}),

                                   ("circulo", SVC, {})], itertools.product([0, 1, 2], repeat=2)):

    clf = clfClass(**params)

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    try:

        fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    except:

        plt.scatter(data[tipo]["X"], data[tipo]["y"])

        plt.plot(np.linspace(0, 120), clf.predict(np.linspace(0, 120)[:, None]))

    plt.title(tipo)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor



gs = gridspec.GridSpec(2, 3)



for (tipo, clfClass), grd  in zip([("regresión", DecisionTreeRegressor),

                                   ("regresión2", DecisionTreeRegressor),

                                   ("iris", DecisionTreeClassifier),

                                   ("xor", DecisionTreeClassifier),

                                   ("circulo", DecisionTreeClassifier)], itertools.product([0, 1, 2], repeat=2)):

    clf = clfClass()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    try:

        fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    except:

        plt.scatter(data[tipo]["X"], data[tipo]["y"])

        plt.plot(np.linspace(0, 120), clf.predict(np.linspace(0, 120)[:, None]))

    plt.title(tipo)
from sklearn.ensemble  import RandomForestClassifier, RandomForestRegressor



gs = gridspec.GridSpec(2, 3)



for (tipo, clfClass), grd  in zip([("regresión", RandomForestRegressor),

                                   ("regresión2", RandomForestRegressor),

                                   ("iris", RandomForestClassifier),

                                   ("xor", RandomForestClassifier),

                                   ("circulo", RandomForestClassifier)], itertools.product([0, 1, 2], repeat=2)):

    clf = clfClass()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    try:

        fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    except:

        plt.scatter(data[tipo]["X"], data[tipo]["y"])

        plt.plot(np.linspace(0, 120), clf.predict(np.linspace(0, 120)[:, None]))

    plt.title(tipo)
from lightgbm import LGBMClassifier, LGBMRegressor



gs = gridspec.GridSpec(2, 3)



for (tipo, clfClass), grd  in zip([("regresión", LGBMRegressor),

                                   ("regresión2", LGBMRegressor),

                                   ("iris", LGBMClassifier),

                                   ("xor", LGBMClassifier),

                                   ("circulo", LGBMClassifier)], itertools.product([0, 1, 2], repeat=2)):

    clf = clfClass()

    clf.fit(data[tipo]["X"], data[tipo]["y"])

    ax = plt.subplot(gs[grd[0], grd[1]])

    try:

        fig = plot_decision_regions(X=data[tipo]["X"], y=data[tipo]["y"], clf=clf, legend=2)

    except:

        plt.scatter(data[tipo]["X"], data[tipo]["y"])

        plt.plot(np.linspace(0, 120), clf.predict(np.linspace(0, 120)[:, None]))

    plt.title(tipo)
data = pd.read_csv("../input/titanic/train.csv", index_col=["PassengerId"])

data
data.dtypes
data.isnull().sum()
for c in data.select_dtypes("O"):

    data[c] = data[c].astype("category")
data["NumFam"] = data.SibSp + data.Parch
data
pd.crosstab(data.NumFam, data.Survived)
pd.crosstab(data.NumFam, data.Survived).apply(lambda x: x / x.sum(), axis=1)
pd.crosstab(data.Pclass, data.Survived).apply(lambda x: x / x.sum(), axis=1)
pd.crosstab(data.NumFam, data.Pclass, values=data.Survived, aggfunc=len)
pd.crosstab(data.NumFam, data.Pclass, values=data.Survived, aggfunc=np.mean)
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import roc_auc_score



X_train, X_test, y_train, y_test = train_test_split(data.drop("Survived", axis=1), 

                                                      data.Survived, test_size=0.1, random_state=2)

kf = KFold(n_splits=5)

folds = [(X_train.iloc[train_idx].index, X_train.iloc[valid_idx].index)

         for train_idx, valid_idx in kf.split(X_train)]



num_leaves = list(range(10, 39, 3))



res = pd.DataFrame([], index=[str(d) for d in num_leaves],

                   columns=["fold_" + str(i) for i in range(len(folds))] + ["ensamble"])



for nl in num_leaves:

    test_probs = []

    for i, (train_idx, valid_idx) in enumerate(folds):

        print("doing fold {0} of depth {1}".format(i + 1, str(nl)))

        Xt = X_train.loc[train_idx]

        yt = y_train.loc[train_idx]



        Xv = X_train.loc[valid_idx]

        yv = y_train.loc[valid_idx]



        learner = LGBMClassifier(n_estimators=10000, num_leaves=nl)

        learner.fit(Xt, yt, early_stopping_rounds=10, eval_metric="auc",

                    eval_set=[(Xt, yt), (Xv, yv)], verbose=3)

        probs = pd.Series(learner.predict_proba(X_test)[:, -1],

                          index=X_test.index, name="fold_" + str(i))

        test_probs.append(probs)

        res.loc[str(nl), "fold_" + str(i)] = roc_auc_score(y_test, probs)

        

    test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

    res.loc[str(nl), "ensamble"] = roc_auc_score(y_test, test_probs)
res
res.var().sort_values()
pd.Series(learner.feature_importances_, index=X_train.columns).sort_values(ascending=False)