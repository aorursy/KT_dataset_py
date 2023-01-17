import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score

from sklearn.linear_model import Ridge, Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from scipy.stats import geom, uniform

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df.head()
raw_X = df.loc[:,'bedrooms':]

raw_X.head()
X = raw_X.copy()

for col in ['waterfront', 'view', 'condition', 'grade']:

    one_hot = pd.get_dummies(X[col], prefix=col)

    X = X.drop(col, axis=1)

    X = X.join(one_hot)

X.head()
y = df['price']

y.head()
inner_cv = KFold(n_splits=2, shuffle=True, random_state=0)

outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

def print_score(s): print(f'Score mean: {s.mean()}')

results = []
params = {

    'alpha': [0.1, 1, 10]

}

lr_gs = GridSearchCV(Ridge(), params, cv=inner_cv)

score = cross_val_score(lr_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('Ridge Regression', score))

print_score(score)
params = {

    'alpha': [0.1, 1, 10]

}

lr_gs = GridSearchCV(Lasso(), params, cv=inner_cv)

score = cross_val_score(lr_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('Lasso Regression', score))

print_score(score)
params = {

    'p__degree': [2, 3],

    'r__alpha': [0.1, 1, 10]

}

pr = Pipeline(steps=[('p', PolynomialFeatures()), ('r', Ridge())])

pr_gs = GridSearchCV(pr, param_grid=params, cv=inner_cv)

score = cross_val_score(pr_gs, raw_X, y, cv=outer_cv, n_jobs=4)

results.append(('Polynomial Regression', score))

print_score(score)
params = {

    'n_neighbors': [3, 5, 10, 15],

    'weights': ['uniform', 'distance']

}

knn_gs = GridSearchCV(KNeighborsRegressor(), params, cv=inner_cv)

score = cross_val_score(knn_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('k-NN', score))

print_score(score)
params = {

    'n_estimators': [10, 50, 100, 250],

    'max_samples': [0.5, 0.75],

    'max_features': [0.5, 0.75, 1.0]

}

rf_gs = GridSearchCV(BaggingRegressor(), params, cv=inner_cv)

score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('Bagging', score))

print_score(score)
params = {

    'n_estimators': [10, 50, 100, 250],

    'max_samples': [0.5, 0.75],

    'max_features': [0.5, 0.75, 1.0]

}

rf_gs = GridSearchCV(RandomForestRegressor(), params, cv=inner_cv)

score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('Random Forest', score))

print_score(score)
params = {

    'n_estimators': [10, 50, 100, 250],

    'max_samples': [0.5, 0.75, 1.0],

    'max_features': [0.5, 0.75, 1.0]

}

rf_gs = GridSearchCV(ExtraTreesRegressor(bootstrap=True), params, cv=inner_cv)

score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('ERT', score))

print_score(score)
params = {

    'n_estimators': [10, 50, 100, 250],

    'learning_rate': [0.1, 1.0, 10.0],

    'loss': ['linear', 'square', 'exponential']

}

rf_gs = GridSearchCV(AdaBoostRegressor(), params, cv=inner_cv)

score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('AdaBoost', score))

print_score(score)
params = {

    'n_estimators': [10, 50, 100, 250],

    'learning_rate': [0.1, 1.0, 10.0],

    'loss': ['ls', 'lad', 'huber', 'quantile']

}

rf_gs = GridSearchCV(GradientBoostingRegressor(), params, cv=inner_cv)

score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('Gradient Boosting', score))

print_score(score)
params = {

    'n_estimators': [10, 50, 100, 250],

    'learning_rate': [0.1, 0.5, 1.0]

}

rf_gs = GridSearchCV(XGBRegressor(), params, cv=inner_cv)

score = cross_val_score(rf_gs, X, y, cv=outer_cv, n_jobs=4)

results.append(('XGBoost', score))

print_score(score)
names = list(map(lambda x: x[0], results))

data = list(map(lambda x: x[1], results))

fig, ax = plt.subplots(figsize=(15, 5))

plt.setp(ax.get_xticklabels(), rotation=45)

sns.boxplot(x=names, y=data)

plt.show()
inner_cv = KFold(n_splits=2, shuffle=True, random_state=0)

outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

estimator_results = []
params = {

    'base_estimator__kernel': ['poly', 'rbf', 'sigmoid'],

    'n_estimators': [10, 25],

    'max_samples': [0.5, 0.75],

    'max_features': [0.5, 0.75, 1.0]

}

rf_svm_gs = GridSearchCV(BaggingRegressor(base_estimator=SVR()), params, cv=inner_cv)

score = cross_val_score(rf_svm_gs, raw_X, y, cv=outer_cv, n_jobs=4)

estimator_results.append(('SVM', score))

print_score(score)
params = {

    'base_estimator__p__degree': [2, 3],

    'base_estimator__r__alpha': [0.1, 1, 10],

    'n_estimators': [10, 25],

    'max_samples': [0.5, 0.75],

    'max_features': [0.5, 0.75, 1.0]

}

pr = Pipeline(steps=[('p', PolynomialFeatures()), ('r', Ridge())])

rf_pr_gs = GridSearchCV(BaggingRegressor(base_estimator=pr), param_grid=params, cv=inner_cv)

score = cross_val_score(rf_pr_gs, raw_X, y, cv=outer_cv, n_jobs=4)

estimator_results.append(('Polynomial Regression', score))

print_score(score)
params = {

    'base_estimator__n_neighbors': [3, 5, 10, 15],

    'base_estimator__weights': ['uniform', 'distance'],

    'n_estimators': [10, 25],

    'max_samples': [0.5, 0.75],

    'max_features': [0.5, 0.75, 1.0]

}

rf_knn_gs = GridSearchCV(BaggingRegressor(base_estimator=KNeighborsRegressor()), params, cv=inner_cv)

score = cross_val_score(rf_knn_gs, X, y, cv=outer_cv, n_jobs=4)

estimator_results.append(('k-NN', score))

print_score(score)
params = {

    'base_estimator__criterion': ['mse', 'friedman_mse', 'mae'],

    'base_estimator__splitter': ['best', 'random'],

    'n_estimators': [10, 25],

    'max_samples': [0.5, 0.75],

    'max_features': [0.5, 0.75, 1.0]

}

rf_dt_gs = GridSearchCV(BaggingRegressor(base_estimator=DecisionTreeRegressor()), params, cv=inner_cv)

score = cross_val_score(rf_dt_gs, X, y, cv=outer_cv, n_jobs=4)

estimator_results.append(('Decision Tree', score))

print_score(score)
names = list(map(lambda x: x[0], estimator_results))

data = list(map(lambda x: x[1], estimator_results))

fig, ax = plt.subplots(figsize=(10, 5))

plt.setp(ax.get_xticklabels(), rotation=45)

sns.boxplot(x=names, y=data)

plt.show()
gs_cv = KFold(n_splits=10, shuffle=True, random_state=0)
params = {

    'n_estimators': geom(p=0.01, loc=50 - 1), # Geometric distribution starting at 50 w/ p=0.01

    'max_samples': uniform(loc=0.5, scale=0.9 - 0.5), # [loc, loc + scale]

    'max_features': uniform(loc=0.5, scale=1.0 - 0.5), # [loc, loc + scale]

    'criterion': ['mse', 'mae']

}

rf_gs = RandomizedSearchCV(ExtraTreesRegressor(bootstrap=True), params, cv=gs_cv, n_iter=20, n_jobs=4)

rf_results = rf_gs.fit(X, y).cv_results_

rf_df = pd.DataFrame.from_dict(rf_results)

rf_df.sort_values('rank_test_score').head()
params = {

    'n_estimators': geom(p=0.01, loc=50 - 1),

    'learning_rate': uniform(loc=0.1, scale=1.0 - 0.1),

    'booster': ['gbtree', 'gblinear', 'dart']

}

xgb_gs = RandomizedSearchCV(XGBRegressor(), params, cv=gs_cv, n_iter=50, n_jobs=4)

xgb_results = xgb_gs.fit(X, y).cv_results_

xgb_df = pd.DataFrame.from_dict(xgb_results)

xgb_df.sort_values('rank_test_score').head()