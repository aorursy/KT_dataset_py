import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); 

from sklearn.metrics import mean_squared_error

from scipy.stats import normaltest

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.tree import export_graphviz

from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head(20).T
df.info()
df_target = df['quality']

df_target
sns.kdeplot(df_target)
df['quality_log'] = np.log(df['quality'])

df_target_log = df['quality_log']

print(df_target)

print(df_target_log)
scaler = StandardScaler()

df_scaled = df.drop('quality', axis = 1)

df2 = df_scaled

df_scaled = df.drop('quality_log', axis = 1)

df3 = df2.drop('quality_log', axis = 1)

df_scaled_fin = scaler.fit_transform(df3)
X_train, X_valid, y_train, y_valid = train_test_split(df_scaled_fin, df_target_log, test_size=0.25, random_state=412)

df_scaled_fin.shape
tree = DecisionTreeRegressor(max_depth=3, random_state=412)

tree.fit(X_train, y_train)

tree
y_pred = tree.predict(X_valid)



print(tree.score(X_valid, y_valid))

print(mean_squared_error(y_valid, y_pred))
kf = KFold(n_splits=5, shuffle=True, random_state=412)

tree = DecisionTreeRegressor(max_depth=3, random_state=412)
tree_params = {'max_depth': np.arange(1, 12)}

tree_grid_depth = GridSearchCV(tree, tree_params, scoring='neg_mean_squared_error', cv=kf, n_jobs = -1)

tree_grid_depth.fit(X_train, y_train)
best_params_depth = tree_grid_depth.best_params_['max_depth']



print(tree_grid_depth.best_estimator_)

print(best_params_depth)

print(tree_grid_depth.best_score_)
tree_params = {'min_samples_split':np.arange(1,12)}

tree = DecisionTreeRegressor(max_depth = best_params_depth)

tree_grid_samples_split = GridSearchCV(tree, tree_params, scoring='neg_mean_squared_error', cv=kf, n_jobs = -1)

tree_grid_samples_split.fit(X_train, y_train)
best_params_samples_split = tree_grid_samples_split.best_params_['min_samples_split']



print(tree_grid_samples_split.best_estimator_)

print(best_params_samples_split)

print(tree_grid_samples_split.best_score_)
tree_params = {'min_samples_leaf':np.arange(1,12)}

tree = DecisionTreeRegressor(max_depth = best_params_depth, min_samples_split = best_params_samples_split)

tree_grid_samples_leaf = GridSearchCV(tree, tree_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

tree_grid_samples_leaf.fit(X_train, y_train)
best_params_samples_leaf = tree_grid_samples_leaf.best_params_['min_samples_leaf']



print(tree_grid_samples_leaf.best_estimator_)

print(best_params_samples_leaf)

print(tree_grid_samples_leaf.best_score_)
tree_params = {'max_features':np.arange(1,12)}

tree = DecisionTreeRegressor(max_depth = best_params_depth, min_samples_split = best_params_samples_split, 

                             min_samples_leaf = best_params_samples_leaf)

tree_grid_max_features = GridSearchCV(tree, tree_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

tree_grid_max_features.fit(X_train, y_train)
best_params_max_features = tree_grid_max_features.best_params_['max_features']



print(tree_grid_max_features.best_estimator_)

print(best_params_max_features)

print(tree_grid_max_features.best_score_)
results_grid_depth = pd.DataFrame(tree_grid_depth.cv_results_)

results_samples_split = pd.DataFrame(tree_grid_samples_split.cv_results_)

results_grid_samples_leaf = pd.DataFrame(tree_grid_samples_leaf.cv_results_)

results_grid_max_features = pd.DataFrame(tree_grid_max_features.cv_results_)
fig, ax = plt.subplots(2, 2, figsize = (10,10))



ax[0,0].set_xlabel("Max depth")

ax[0,0].set_ylabel("Test error")

ax[0,0].plot(results_grid_depth['param_max_depth'], results_grid_depth['mean_test_score']);



ax[0,1].set_xlabel("Min samples split")

ax[0,1].set_ylabel("Test error")

ax[0,1].plot(results_samples_split['param_min_samples_split'], results_samples_split['mean_test_score']);



ax[1,0].set_xlabel("Min samples leaf")

ax[1,0].set_ylabel("Test error")

ax[1,0].plot(results_grid_samples_leaf['param_min_samples_leaf'],results_grid_samples_leaf['mean_test_score']);



ax[1,1].set_xlabel("Max features")

ax[1,1].set_ylabel("Test error")

ax[1,1].plot(results_grid_max_features['param_max_features'], results_grid_max_features['mean_test_score']);
print('Один из вариантов гиперпараметров:')

print('best_params_depth = ', best_params_depth)

print('best_params_samples_split = ', best_params_samples_split)

print('best_params_samples_leaf = ', best_params_samples_leaf)

print('best_params_max_features = ', best_params_max_features)
tree = DecisionTreeRegressor(max_depth = best_params_depth, min_samples_split = best_params_samples_split, min_samples_leaf = best_params_samples_leaf, 

                                  max_features =  best_params_max_features, random_state = 412)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)

mean_squared_error(y_valid, y_pred)
export_graphviz(tree, out_file = 'tree.dot', feature_names = df3.columns)

print(open('tree.dot').read())
features = {'f' + str(i + 1):name for (i, name) in zip(range(len(df3.columns)), df3.columns)}

importances = tree.feature_importances_



indices = np.argsort(importances)[:: -1]

num_to_plot = 10

feature_indices = [ind + 1 for ind in indices[:num_to_plot]]



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
rf = RandomForestRegressor(random_state = 412)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



mean_squared_error(y_valid, y_pred)
rf_params = {'n_estimators': [250, 300, 350, 400, 450, 500]}

rf_grid_estimators = GridSearchCV(rf, rf_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

rf_grid_estimators.fit(X_train, y_train)
best_params_estimators = rf_grid_estimators.best_params_['n_estimators']



print(rf_grid_estimators.best_estimator_)

print(best_params_estimators)

print(rf_grid_estimators.best_score_)
rf = RandomForestRegressor(random_state = 412, n_estimators = best_params_estimators)

rf_params = {'max_depth': np.arange(1, 20)}

rf_grid_depth = GridSearchCV(rf, rf_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

rf_grid_depth.fit(X_train, y_train)
best_params_max_depth = rf_grid_depth.best_params_['max_depth']



print(rf_grid_depth.best_estimator_)

print(best_params_max_depth)

print(rf_grid_depth.best_score_)
rf = RandomForestRegressor(random_state = 412, n_estimators = best_params_estimators, max_depth = best_params_max_depth)

rf_params = {'min_samples_split': np.arange(1, 12)}

rf_grid_split = GridSearchCV(rf, rf_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

rf_grid_split.fit(X_train, y_train)
best_params_min_samples_split = rf_grid_split.best_params_['min_samples_split']



print(rf_grid_split.best_estimator_)

print(best_params_min_samples_split)

print(rf_grid_split.best_score_)
rf = RandomForestRegressor(random_state = 412, n_estimators = best_params_estimators, max_depth = best_params_max_depth, min_samples_split = best_params_min_samples_split)

rf_params = {'min_samples_leaf': np.arange(1, 10)}

rf_grid_leaf = GridSearchCV(rf, rf_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

rf_grid_leaf.fit(X_train, y_train)
best_params_min_samples_leaf = rf_grid_leaf.best_params_['min_samples_leaf']



print(rf_grid_leaf.best_estimator_)

print(best_params_min_samples_leaf)

print(rf_grid_leaf.best_score_)
rf = RandomForestRegressor(random_state = 412, n_estimators = best_params_estimators, max_depth = best_params_max_depth, min_samples_split = best_params_min_samples_split, 

                           min_samples_leaf = best_params_min_samples_leaf)

rf_params = {'max_features': np.arange(1, 12)}

rf_grid_features = GridSearchCV(rf, rf_params, cv=kf, scoring='neg_mean_squared_error', n_jobs = -1)

rf_grid_features.fit(X_train, y_train)
best_params_max_features = rf_grid_features.best_params_['max_features']



print(rf_grid_features.best_estimator_)

print(best_params_max_features)

print(rf_grid_features.best_score_)
results_rf_grid_estimators = pd.DataFrame(rf_grid_estimators.cv_results_)

results_rf_grid_depth = pd.DataFrame(rf_grid_depth.cv_results_)

results_rf_grid_split = pd.DataFrame(rf_grid_split.cv_results_)

results_rf_grid_leaf = pd.DataFrame(rf_grid_leaf.cv_results_)

results_rf_grid_features = pd.DataFrame(rf_grid_features.cv_results_)
fig, ax = plt.subplots(2, 3, figsize = (15,15))



ax[0,0].set_xlabel("N Estimators")

ax[0,0].set_ylabel("Test error")

ax[0,0].plot(results_rf_grid_estimators['param_n_estimators'], results_rf_grid_estimators["mean_test_score"]);



ax[0,1].set_xlabel("Max depth")

ax[0,1].set_ylabel("Test error")

ax[0,1].plot(results_rf_grid_depth['param_max_depth'], results_rf_grid_depth["mean_test_score"]);



ax[0,2].set_xlabel("Min samples split")

ax[0,2].set_ylabel("Test error")

ax[0,2].plot(results_rf_grid_split['param_min_samples_split'],results_rf_grid_split["mean_test_score"]);



ax[1,0].set_xlabel("Min samples leaf")

ax[1,0].set_ylabel("Test error")

ax[1,0].plot(results_rf_grid_leaf['param_min_samples_leaf'], results_rf_grid_leaf['mean_test_score']);



ax[1,1].set_xlabel("Max features")

ax[1,1].set_ylabel("Test error")

ax[1,1].plot(results_rf_grid_features['param_max_features'], results_rf_grid_features['mean_test_score']);
print('Один из вариантов гиперпараметров:')

print('best_params_estimators = ', best_params_estimators)

print('best_params_max_depth = ', best_params_max_depth)

print('best_params_min_samples_split = ', best_params_min_samples_split)

print('best_params_min_samples_leaf = ', best_params_min_samples_leaf)

print('best_params_max_features = ', best_params_max_features)
features = {'f' + str(i + 1):name for (i, name) in zip(range(len(df3.columns)), df3.columns)}

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