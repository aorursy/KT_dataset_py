# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Financial Distress.csv')
data.info()
data.describe()
data.head(5)
data.shape
categories = data['x80'].unique()
print(categories)
print(len(categories))
g = sns.PairGrid(data, x_vars='Time', y_vars=list(data.columns[3:]), hue='Company', size=5)
g = g.map(plt.scatter, alpha=.3)
sns.distplot(data['Time'])
data_corr = data.drop(labels=['Company'], axis=1).corr()
data_corr = data_corr.sort_values(ascending=False, axis=1, by='Financial Distress')
data_corr.head(10)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize = (20,20))
sns.heatmap(data_corr, cmap=cmap)
distressed = [1 if row['Financial Distress'] <= -0.5 else 0 for _, row in data.iterrows()]
data_full = data
data_full['Distressed'] = pd.Series(distressed)
data_full.loc[data_full['Distressed'] == 1, ['Financial Distress', 'Distressed']].head(10)
g = sns.PairGrid(data_full, x_vars='Time', y_vars=list(data.columns[3:]), hue='Distressed', size=5)
g = g.map(plt.scatter, alpha=.3)
from sklearn.model_selection import StratifiedShuffleSplit
SSS = StratifiedShuffleSplit(random_state=10, test_size=.3, n_splits=1)
X = data_full.iloc[:, 3:-1].drop('x80', axis=1)
y = data_full['Distressed'] 
for train_index, test_index in SSS.split(X, y):
    print("CV:", train_index, "HO:", test_index)
    X_cv, X_ho = X.iloc[train_index], X.iloc[test_index]
    y_cv, y_ho = y[train_index], y[test_index]
# X_cv, X_ho, y_cv, y_ho = StratifiedShuffleSplit(data_shuffled.iloc[:, 3:-1], data_shuffled['Distressed'],
#                                                    test_size=0.33, random_state=10)
print('CV distress:', sum(y_cv), '\nHO distress:', sum(y_ho))
data_full['Distressed'].value_counts()
136/3536
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pprint import pprint
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 55, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]
class_weight = ['balanced', None]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}

rf_clsf = RandomForestClassifier(random_state=10, class_weight='balanced')
rf_random = RandomizedSearchCV(estimator = rf_clsf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=10, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
rf_random.fit(X_cv, y_cv)
# print(rf_random.best_score_, '\n', rf_random.cv_results_)
print(rf_random.best_score_)
best_rf_clsf = rf_random.best_estimator_
best_rf_clsf.fit(X_cv, y_cv)
print(recall_score(y_ho, best_rf_clsf.predict(X_ho)),
      precision_score(y_ho, best_rf_clsf.predict(X_ho)),
      f1_score(y_ho, best_rf_clsf.predict(X_ho)))
print(sum(y_ho), sum(best_rf_clsf.predict(X_ho)), sum(y_ho.multiply(best_rf_clsf.predict(X_ho))))
lr_penalty = ['l1']
lr_class_weight = ['balanced', None]
lr_C = [0.001, 0.01, 0.1, 1, 10]
#lr_max_iter = [int(x) for x in np.linspace(100, 1000, num = 100)]
lr_solver = ['liblinear', 'saga']

lr_grid = {'penalty': lr_penalty,
           'class_weight': lr_class_weight,
           'C': lr_C,
           #'max_iter': lr_max_iter,
           'solver': lr_solver}
lr_clsf = LogisticRegression(random_state=10, max_iter=1000)
lr_grid = GridSearchCV(estimator = lr_clsf, param_grid = lr_grid, cv = 3, verbose=2, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
lr_grid.fit(X_cv, y_cv)
lr_grid.best_score_
best_lr_clsf = lr_grid.best_estimator_
best_lr_clsf.fit(X_cv, y_cv)
print(recall_score(y_ho, best_lr_clsf.predict(X_ho)),
      precision_score(y_ho, best_lr_clsf.predict(X_ho)),
      f1_score(y_ho, best_lr_clsf.predict(X_ho)))
print(sum(y_ho), sum(best_lr_clsf.predict(X_ho)), sum(y_ho.multiply(best_lr_clsf.predict(X_ho))))
xgb_learning_rate = [x for x in np.linspace(start = 0.001, stop = 0.1, num = 10)]
xgb_n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
xgb_booster = ['gbtree', 'dart']
xgb_colsample_bytree = [0.4, 0.6, 0.8, 1.0]
xgb_colsample_bylevel = [0.5, 0.75, 1.0]
xgb_scale_pos_weight = [(len(y_cv) - sum(y_cv))/sum(y_cv)]
xgb_min_child_weight = [1]
xgb_subsample = [0.5, 1.0]


random_grid = {'learning_rate': xgb_learning_rate,
               'n_estimators': xgb_n_estimators,
               'booster': xgb_booster,
               'colsample_bytree': xgb_colsample_bytree,
               'colsample_bylevel': xgb_colsample_bylevel,
               'scale_pos_weight': xgb_scale_pos_weight,
               'min_child_weight': xgb_min_child_weight,
               'subsample': xgb_subsample}
xgb_clsf = xgb.XGBClassifier(random_state=10)
xgb_random = RandomizedSearchCV(estimator = xgb_clsf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=10, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
xgb_random.fit(X_cv, y_cv)
#xgb_clsf.fit(X_train, y_train)
xgb_random.best_score_
best_xgb_clsf = xgb_random.best_estimator_
best_xgb_clsf.fit(X_cv, y_cv)
print(recall_score(y_ho, best_xgb_clsf.predict(X_ho)),
      precision_score(y_ho, best_xgb_clsf.predict(X_ho)),
      f1_score(y_ho, best_xgb_clsf.predict(X_ho)))
print(sum(y_ho), sum(best_xgb_clsf.predict(X_ho)), sum(y_ho.multiply(best_xgb_clsf.predict(X_ho))))
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
X = data_full.iloc[:, 3:-1]
y = data_full['Distressed'] 

# data_scaled = pd.concat([data_full['x80'],data_scaled], axis=1)
# enc = OneHotEncoder(n_values=len(X['x80'].unique()), categorical_features=X.columns.get_loc("x80"))
X_encoded = pd.get_dummies(X, columns=['x80'], prefix='x80_')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled = pd.DataFrame(X_scaled, index=X_encoded.index, columns=X_encoded.columns)
#data_scaled['x80'] = data_full['x80'].values
for train_index, test_index in SSS.split(X_scaled, y):
    print("CV:", train_index, "HO:", test_index)
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
sm = SMOTE(random_state=10)
X_train, y_train = sm.fit_sample(X_train, y_train)
# X_train = pd.DataFrame(X_train, index=X_encoded.index, columns=X_encoded.columns)
# y_train = pd.DataFrame(y, index=y.index, columns=y.columns)

y_test
rf_n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
rf_max_features = ['auto', 'sqrt']
rf_max_depth = [int(x) for x in np.linspace(50, 100, num = 10)]
rf_max_depth.append(None)
rf_min_samples_split = [2, 5, 10]
rf_min_samples_leaf = [1, 2, 3, 4]
rf_bootstrap = [True, False]
rf_class_weight = ['balanced', None]

rf_random_grid = {'n_estimators': rf_n_estimators,
               'max_features': rf_max_features,
               'max_depth': rf_max_depth,
               'min_samples_split': rf_min_samples_split,
               'min_samples_leaf': rf_min_samples_leaf,
               'bootstrap': rf_bootstrap,
               'class_weight': rf_class_weight}
rf_clsf = RandomForestClassifier(random_state=10)
rf_random_2 = RandomizedSearchCV(estimator = rf_clsf, param_distributions = rf_random_grid, n_iter = 10, cv = 3, verbose=2, random_state=10, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
rf_random_2.fit(X_train, y_train)
best_rf_clsf = rf_random_2.best_estimator_
best_rf_clsf.fit(X_train, y_train)
print(rf_random_2.best_score_)
print(recall_score(y_test, best_rf_clsf.predict(X_test)),
      precision_score(y_test, best_rf_clsf.predict(X_test)),
      f1_score(y_test, best_rf_clsf.predict(X_test)))
print(sum(y_test), sum(best_rf_clsf.predict(X_test)), sum(y_test.multiply(best_rf_clsf.predict(X_test))))
lr_penalty = ['l1', 'l2']
lr_class_weight = ['balanced', None]
lr_C = [0.1, 1, 10, 100]
#lr_max_iter = [int(x) for x in np.linspace(100, 1000, num = 100)]
lr_solver = ['liblinear', 'saga']

lr_grid = {'penalty': lr_penalty,
           'class_weight': lr_class_weight,
           'C': lr_C,
           #'max_iter': lr_max_iter,
           'solver': lr_solver}
lr_clsf = LogisticRegression(random_state=10, max_iter=2000)
lr_grid_2 = GridSearchCV(estimator = lr_clsf, param_grid = lr_grid, cv = 3, verbose=2, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
lr_grid_2.fit(X_train, y_train)
best_lr_clsf = lr_grid_2.best_estimator_
best_lr_clsf.fit(X_train, y_train)
print(lr_grid_2.best_score_)
print(recall_score(y_test, best_lr_clsf.predict(X_test)),
      precision_score(y_test, best_lr_clsf.predict(X_test)),
      f1_score(y_test, best_lr_clsf.predict(X_test)))
print(sum(y_test), sum(best_lr_clsf.predict(X_test)), sum(y_test.multiply(best_lr_clsf.predict(X_test))))
xgb_learning_rate = [x for x in np.linspace(start = 0.001, stop = 0.1, num = 10)]
xgb_n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
xgb_booster = ['gbtree', 'dart']
xgb_colsample_bytree = [0.4, 0.6, 0.8, 1.0]
xgb_colsample_bylevel = [0.5, 0.75, 1.0]
xgb_scale_pos_weight = [(len(y_cv) - sum(y_cv))/sum(y_cv)]
xgb_min_child_weight = [1]
xgb_subsample = [0.5, 1.0]


random_grid = {'learning_rate': xgb_learning_rate,
               'n_estimators': xgb_n_estimators,
               'booster': xgb_booster,
               'colsample_bytree': xgb_colsample_bytree,
               'colsample_bylevel': xgb_colsample_bylevel,
               'scale_pos_weight': xgb_scale_pos_weight,
               'min_child_weight': xgb_min_child_weight,
               'subsample': xgb_subsample}
xgb_clsf = xgb.XGBClassifier(random_state=10)
xgb_random_2 = RandomizedSearchCV(estimator = xgb_clsf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=10, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
xgb_random_2.fit(X_train, y_train)
best_xgb_clsf = xgb_random_2.best_estimator_
best_xgb_clsf.fit(X_train, y_train)
print(xgb_random_2.best_score_)
print(recall_score(y_test, best_xgb_clsf.predict(X_test.values)),
      precision_score(y_test, best_xgb_clsf.predict(X_test.values)),
      f1_score(y_test, best_xgb_clsf.predict(X_test.values)))
print(sum(y_test), sum(best_xgb_clsf.predict(X_test.values)), sum(y_test.multiply(best_xgb_clsf.predict(X_test.values))))




