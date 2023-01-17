import numpy as np

from matplotlib import pyplot

from pandas import read_csv

import pandas as pd

from sklearn.preprocessing import Imputer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor
train = read_csv('../input/train.csv')

train.shape
train.head()
X_train = train.loc[:,'MSSubClass':'SaleCondition']

X_train = pd.get_dummies(X_train)

imp = Imputer(missing_values=0, strategy='mean', axis=0)

imp.fit(X_train)

X_train = imp.transform(X_train)

Y_train = train.loc[:,"SalePrice"]

X_train[0:5]
X_train.shape
kbest = SelectKBest(chi2, k=50)

X_new = kbest.fit_transform(X_train,Y_train)

Y_new = np.log1p(Y_train)
num_folds = 10

seed = 7

scoring = 'neg_mean_squared_error'



models = []

models.append(('LR', LinearRegression()))

models.append(('Lasso', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_new, Y_new, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {} ({})".format(name, cv_results.mean(),  cv_results.std()))
fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))

pipelines.append(('ScaledLasso', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART',DecisionTreeRegressor())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),  ('SVR', SVR())])))



results= []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_new, Y_new, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))
fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
ensembles = []

ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))

ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))



results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_new, Y_new, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))

    
fig = pyplot.figure()

fig.suptitle('Scaled Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()