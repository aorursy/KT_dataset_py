# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv')
df.head()
df.describe().T
df.info()
df1 = df.drop('url', axis=1)
snsplot = sns.kdeplot(df['shares'], shade=True)
fig = snsplot.get_figure()
df1['shares_log'] = np.log(df1['shares'])
df2 = df1.drop('shares', axis=1)
X = df2.drop('shares_log', axis=1)
y = df2['shares_log'] 
X.head()
scaler = StandardScaler()
X_new = scaler.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, test_size = 0.25, random_state = 19)
tree = DecisionTreeRegressor(max_depth = 3, random_state = 19)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
mean_squared_error(y_valid, y_pred)
from sklearn.metrics import r2_score
r2_score(y_valid, y_pred)
tree.score(X_valid, y_valid)
kf = KFold(n_splits = 5, shuffle = True, random_state = 30)
tree_params_depth = {'max_depth': np.arange(2, 11)}

tree_grid_depth = GridSearchCV(tree, tree_params_depth, cv=5, scoring='r2', n_jobs = -1)
tree_grid_depth.fit(X_train, y_train)
print(tree_grid_depth.best_params_)
print(tree_grid_depth.best_score_)
tree = DecisionTreeRegressor(max_depth = 3)
tree_params_split = {'min_samples_split': np.arange(2, 21)}

tree_grid_split = GridSearchCV(tree, tree_params_split , cv=5, scoring='r2', n_jobs = -1)
tree_grid_split.fit(X_train, y_train)
print(tree_grid_split.best_params_)
print(tree_grid_split.best_score_)
tree = DecisionTreeRegressor(max_depth = 3, min_samples_split = 14)
tree_params_leaf = {'min_samples_leaf': np.arange(2, 21)}

tree_grid_leaf = GridSearchCV(tree, tree_params_leaf, cv=5, scoring='r2', n_jobs = -1)
tree_grid_leaf.fit(X_train, y_train)
print(tree_grid_leaf.best_params_)
print(tree_grid_leaf.best_score_)
tree = DecisionTreeRegressor(max_depth = 3, min_samples_split = 14, min_samples_leaf = 16)
tree_params_features = {'max_features': np.arange(2, 21)}

tree_grid_features = GridSearchCV(tree, tree_params_features, cv=5, scoring='r2', n_jobs = -1)
tree_grid_features.fit(X_train, y_train)
print(tree_grid_features.best_params_)
print(tree_grid_features.best_score_)
fig, ax = plt.subplots(2, 2, figsize = (10,10))

ax[0,0].set_xlabel("Max depth")
ax[0,0].set_ylabel("Score")
ax[0,0].plot(tree_params_depth['max_depth'], tree_grid_depth.cv_results_["mean_test_score"]);

ax[0,1].set_xlabel("Min samples split")
ax[0,1].set_ylabel("Score")
ax[0,1].plot(tree_params_split["min_samples_split"], tree_grid_split.cv_results_["mean_test_score"]);

ax[1,0].set_xlabel("Min samples leaf")
ax[1,0].set_ylabel("Score")
ax[1,0].plot(tree_params_leaf["min_samples_leaf"],tree_grid_leaf.cv_results_["mean_test_score"]);

ax[1,1].set_xlabel("Max features")
ax[1,1].set_ylabel("Score")
ax[1,1].plot(tree_params_features["max_features"], tree_grid_features.cv_results_["mean_test_score"]);
tree = DecisionTreeRegressor(max_depth = 3, min_samples_split = 14, min_samples_leaf = 16, max_features =  15, random_state = 19)
tree.fit(X_train, y_train)
r2_score(y_valid, y_pred)
export_graphviz(tree, out_file = 'tree.dot', feature_names = X.columns)
print(open('tree.dot').read())
features = {'f' + str(i + 1):name for (i, name) in zip(range(len(df2.columns)), df2.columns)}
importances = tree.feature_importances_

indices = np.argsort(importances)[:: -1]
num_to_plot = 10
feature_indices = [ind + 1 for ind in indices[:num_to_plot]]

print("Feature ranking:")
for f in range(num_to_plot):
    print(f + 1, features["f" + str(feature_indices[f])], importances[indices[f]])

plt.figure(figsize=(15,5))
plt.title("Feature importances")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
               color=([str(i / float(num_to_plot +  1)) for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f" + str(i)]) for i in feature_indices]);
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 19)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)

mean_squared_error(y_valid, y_pred)
r2_score(y_valid, y_pred)
rf_params_estimators = {'n_estimators': [300, 350, 400, 450, 500, 550]}
rf_grid_estimators = GridSearchCV(rf, rf_params_estimators, cv=kf, scoring='r2', n_jobs = -1)
rf_grid_estimators.fit(X_train, y_train)
print(rf_grid_estimators.best_params_)
print(rf_grid_estimators.best_score_)
rf = RandomForestRegressor(random_state = 19, n_estimators = 500)
rf_params_depth = {'max_depth': np.arange(2, 11)}
rf_grid_depth = GridSearchCV(rf, rf_params_depth, cv=kf, scoring='r2', n_jobs = -1)
rf_grid_depth.fit(X_train, y_train)
print(rf_grid_depth.best_params_)
print(rf_grid_depth.best_score_)
rf = RandomForestRegressor(random_state = 19, n_estimators = 500, max_depth = 9)
rf_params_split = {'min_samples_split': np.arange(20, 31)}
rf_grid_split = GridSearchCV(rf, rf_params_split, cv=kf, scoring='r2', n_jobs = -1)
rf_grid_split.fit(X_train, y_train)
print(rf_grid_split.best_params_)
print(rf_grid_split.best_score_)
rf = RandomForestRegressor(random_state = 19, n_estimators = 500, max_depth = 9, min_samples_split = 29)
rf_params_leaf = {'min_samples_leaf': np.arange(18, 31)}
rf_grid_leaf = GridSearchCV(rf, rf_params_leaf, cv=kf, scoring='r2', n_jobs = -1)
rf_grid_leaf.fit(X_train, y_train)
print(rf_grid_leaf.best_params_)
print(rf_grid_leaf.best_score_)
rf = RandomForestRegressor(random_state = 19, n_estimators = 500, max_depth = 9, min_samples_split = 29, min_samples_leaf = 20)
rf_params_features = {'max_features': np.arange(18, 31)}
rf_grid_features = GridSearchCV(rf, rf_params_features, cv=kf, scoring='r2', n_jobs = -1)
rf_grid_features.fit(X_train, y_train)
print(rf_grid_features.best_params_)
print(rf_grid_features.best_score_)
fig, ax = plt.subplots(2, 3, figsize = (15,15))

ax[0,0].set_xlabel("Max depth")
ax[0,0].set_ylabel("Score")
ax[0,0].plot(rf_params_depth['max_depth'], rf_grid_depth.cv_results_["mean_test_score"]);

ax[0,1].set_xlabel("Min samples split")
ax[0,1].set_ylabel("Score")
ax[0,1].plot(rf_params_split["min_samples_split"], rf_grid_split.cv_results_["mean_test_score"]);

ax[0,2].set_xlabel("Min samples leaf")
ax[0,2].set_ylabel("Score")
ax[0,2].plot(rf_params_leaf["min_samples_leaf"],rf_grid_leaf.cv_results_["mean_test_score"]);

ax[1,0].set_xlabel("Max features")
ax[1,0].set_ylabel("Score")
ax[1,0].plot(rf_params_features["max_features"], rf_grid_features.cv_results_["mean_test_score"]);

ax[1,1].set_xlabel("N Estimators")
ax[1,1].set_ylabel("Score")
ax[1,1].plot(rf_params_estimators["n_estimators"], rf_grid_estimators.cv_results_["mean_test_score"]);
features = {'f' + str(i + 1):name for (i, name) in zip(range(len(df2.columns)), df2.columns)}
importances = tree.feature_importances_

indices = np.argsort(importances)[:: -1]
num_to_plot = 10
feature_indices = [ind + 1 for ind in indices[:num_to_plot]]

print("Feature ranking:")
for f in range(num_to_plot):
    print(f + 1, features["f" + str(feature_indices[f])], importances[indices[f]])
plt.figure(figsize=(15,5))
plt.title("Feature importances")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
               color=([str(i / float(num_to_plot +  1)) for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f" + str(i)]) for i in feature_indices]);