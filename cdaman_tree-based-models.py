%matplotlib inline

import matplotlib

import seaborn as sns

sns.set()

matplotlib.rcParams['figure.dpi'] = 144
import numpy as np

import matplotlib.pyplot as plt
import graphviz

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier, export_graphviz



# load data set

data = load_iris()

X = data['data']

y = data['target']



# train decision tree

tree = DecisionTreeClassifier(max_depth=3,)

tree.fit(X, y)



# visual tree

graphviz.Source(export_graphviz(tree, 

                                out_file=None,

                                feature_names=data['feature_names'],

                                class_names=data['target_names']))
from ipywidgets import interact, IntSlider



def iris_tree(depth=1):

    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.viridis)

    

    if depth >= 1:

        plt.hlines(0.8, 0.8, 7, linewidth=2)

    if depth >= 2:

        plt.hlines(1.75, 0.8, 7, linewidth=2)

    if depth >= 3:

        plt.vlines(4.85, 1.75, 2.6, linewidth=2)

        plt.vlines(4.95, 0.8, 1.74, linewidth=2)



    plt.xlabel('Petal Length (cm)')

    plt.ylabel('Petal Width (cm)')

    plt.xlim([0.8, 7])

    plt.ylim([0, 2.6])

    

depth_slider = IntSlider(value=0, min=0, max=3, step=1, description='depth')

interact(iris_tree, depth=depth_slider);
p = np.linspace(1E-6, 1-1E-6, 100)

gini = p*(1-p) + (1-p)*p



plt.plot(p, gini)

plt.xlabel('$p$')

plt.ylabel('Gini');
from sklearn.datasets import fetch_california_housing

from sklearn.tree import DecisionTreeRegressor



# load data set

data = fetch_california_housing()

X = data['data']

y = data['target']



# train decision tree

tree = DecisionTreeRegressor(max_depth=3)

tree.fit(X, y)



# visual tree

graphviz.Source(export_graphviz(tree, 

                                out_file=None,

                                feature_names=data['feature_names']))
from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split



X, y = make_moons(n_samples=250, noise=0.25, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



def tree_decision_boundary(max_depth=5, min_samples_leaf=2):

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    tree.fit(X_train, y_train)

    accuracy = tree.score(X_test, y_test)

    

    X1, X2 = np.meshgrid(np.linspace(-2, 3), np.linspace(-2, 2))

    y_proba = tree.predict_proba(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))))[:, 1]

    plt.contourf(X1, X2, y_proba.reshape(50, 50),  16, cmap=plt.cm.bwr, alpha=0.75)

    plt.colorbar()



    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='white', cmap=plt.cm.bwr)

    plt.xlabel('$x_1$')

    plt.ylabel('$x_2$')

    plt.title('accuracy: {}'.format(accuracy));



depth_slider = IntSlider(min=1, max=40, step=1, description='max depth')

min_samples_leaf_slider = IntSlider(min=1, max=20, step=1, description='min leaf size')

interact(tree_decision_boundary, max_depth=depth_slider, min_samples_leaf=min_samples_leaf_slider);
from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



X, y = make_regression(n_samples=1000, n_features=100, n_informative=20, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



def rf_mse(max_features='sqrt', n_max=50):

    """Generate mean squared errors for growing random forest."""

    

    rgr = RandomForestRegressor(max_features=max_features,

                                max_depth=8, n_estimators=1, 

                                warm_start=True, 

                                random_state=0)

    mse = np.zeros(n_max)



    for n in range(1, n_max):

        rgr.set_params(n_estimators=n)

        rgr.fit(X_train, y_train)

        mse[n-1] = mean_squared_error(y_test, rgr.predict(X_test))



    return mse



for param in ('sqrt', 'log2'):

    mse = rf_mse(max_features=param)

    plt.plot(mse[:-1])



plt.xlabel('number of trees')

plt.ylabel('mean squared error')

plt.legend(['sqrt', 'log2', 'all']);




from sklearn.ensemble import GradientBoostingRegressor



def gb_mse(learning_rate=1.0, subsample=1.0, n_max=80):

    """Generate mean squared errors for growing gradient boosting trees."""

    

    rgr = GradientBoostingRegressor(learning_rate=learning_rate,

                                    subsample=subsample,

                                    max_depth=2, 

                                    n_estimators=1, 

                                    warm_start=True, 

                                    random_state=0)

    mse = np.zeros(n_max)



    for n in range(1, n_max):

        rgr.set_params(n_estimators=n)

        rgr.fit(X_train, y_train)

        mse[n-1] = mean_squared_error(y_test, rgr.predict(X_test))



    return mse



def gen_legend_str(hparams):

    """Generate strings for legend in plot."""

    

    base_str = 'learning rate: {} subsample: {}'

    

    return [base_str.format(d['learning_rate'], d['subsample']) for d in hparams]



hparams = ({'learning_rate': 1.0, 'subsample': 1.0},

           {'learning_rate': 1.0, 'subsample': 0.5},

           {'learning_rate': 0.75, 'subsample': 0.5},

           {'learning_rate': 0.5, 'subsample': 0.5})



for kwargs in hparams:

    mse = gb_mse(**kwargs)

    plt.plot(mse[:-1])



legend_strs = gen_legend_str(hparams)

plt.xlabel('number of trees')

plt.ylabel('mean squared error')

plt.legend(legend_strs);



import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# load data set

data = load_iris()

X = data['data']

y = data['target']

feature_names = data['feature_names']



# tune random forest

tree = RandomForestClassifier(n_estimators=20, random_state=0)

param_grid = {'max_depth': range(2, 10), 'min_samples_split': [2, 4, 6, 8, 10]}

grid_search = GridSearchCV(tree, param_grid, cv=3, n_jobs=2, verbose=1, iid=True)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_



# plot feature importance

df = pd.DataFrame({'importance': best_model.feature_importances_}, index=feature_names)

df.plot.bar();