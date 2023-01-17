import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set();
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv') 
df.head()
df.info()#просмотр колонок
df.describe().T
df['shares_log'] = np.log(df['shares'])
df_1 = df.drop(['shares', 'url'], axis=1) # удалим shares
df_1
X = df_1.drop('shares_log', axis = 1)
y = df_1['shares_log']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_new = scaler.fit_transform(X)
X_new
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, test_size=0.25, random_state=42)
# test_size --- доля исходных данных, которую оставляем для валидации, random_state --- произвольное целое число, для воспроизводимости случайных результатов

from sklearn.tree import DecisionTreeRegressor 
#Обучение
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
y_pred
#Предсказание
from sklearn.metrics import mean_squared_error

mean_squared_error(y_valid, y_pred)
tree.score(X_valid, y_valid)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42) 
# Кросс-валидация и подбор гиперпараметров
from sklearn.model_selection import GridSearchCV

tree_params_max_depth = {'max_depth': np.arange(3, 20)}
tree_grid_max_depth = GridSearchCV(tree, tree_params_max_depth, cv=kf, scoring='explained_variance', n_jobs = -1)
tree_grid_max_depth.fit(X_train, y_train)
max_depth0 = list(tree_grid_max_depth.best_params_.values())[0]
print("max глубина: ", max_depth0)
print("Найлучшая оценка качества модели: ", tree_grid_max_depth.best_score_)
tree_grid_max_depth.best_estimator_
tree = DecisionTreeRegressor(max_depth=max_depth0)
tree_params_min_samples_split = {'min_samples_split': np.arange(3, 20)}
tree_grid_min_samples_split = GridSearchCV(tree, tree_params_min_samples_split, cv=kf, scoring='explained_variance', n_jobs = -1)
tree_grid_min_samples_split.fit(X_train, y_train)
min_samples_split0 = list(tree_grid_min_samples_split.best_params_.values())[0]
print("min разбиения во внутренней вершине: ", min_samples_split0)
print("Найлучшая оценка качества модели: ", tree_grid_min_samples_split.best_score_)
tree_grid_min_samples_split.best_estimator_
tree = DecisionTreeRegressor(max_depth=max_depth0, min_samples_split=min_samples_split0)
tree_params_min_samples_leaf = {'min_samples_leaf': np.arange(3, 20)}
tree_grid_min_samples_leaf = GridSearchCV(tree, tree_params_min_samples_leaf, cv=kf, scoring='explained_variance', n_jobs = -1) 
tree_grid_min_samples_leaf.fit(X_train, y_train)
min_samples_leaf0 = list(tree_grid_min_samples_leaf.best_params_.values())[0]
print("min объектов в листе: ", min_samples_leaf0)
print("Найлучшая оценка качества модели: ", tree_grid_min_samples_leaf.best_score_)
tree_grid_min_samples_leaf.best_estimator_
tree = DecisionTreeRegressor(max_depth=max_depth0, min_samples_split=min_samples_split0, min_samples_leaf=min_samples_leaf0)
tree_params_max_features = {'max_features': np.arange(3, 20)}
tree_grid_max_features = GridSearchCV(tree, tree_params_max_features, cv=kf, scoring='explained_variance', n_jobs = -1) 
tree_grid_max_features.fit(X_train, y_train)
max_features0 = list(tree_grid_max_features.best_params_.values())[0]
print("max, рассматриваемых при поиске лучшего разбиения: ", max_features0)
print("Найлучшая оценка качества модели: ", tree_grid_max_features.best_score_)
tree_grid_max_features.best_estimator_
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, figsize = (20,10))

ax[0, 0].plot(tree_params_max_depth['max_depth'], tree_grid_max_depth.cv_results_['mean_test_score']) 
ax[0, 0].set_xlabel('max_depth')
ax[0, 0].set_ylabel('Mean accuracy on test set')

ax[0, 1].plot(tree_params_min_samples_leaf['min_samples_leaf'], tree_grid_min_samples_leaf.cv_results_['mean_test_score'])
ax[0, 1].set_xlabel('min_samples_leaf')
ax[0, 1].set_ylabel('Mean accuracy on test set')

ax[1, 0].plot(tree_params_min_samples_split['min_samples_split'], tree_grid_min_samples_split.cv_results_['mean_test_score']) 
ax[1, 0].set_xlabel('min_samples_split')
ax[1, 0].set_ylabel('Mean accuracy on test set')

ax[1, 1].plot(tree_params_max_features['max_features'], tree_grid_max_features.cv_results_['mean_test_score'])
ax[1, 1].set_xlabel('max_features')
ax[1, 1].set_ylabel('Mean accuracy on test set')
print(max_depth0)
print(min_samples_split0)
print(min_samples_leaf0)
print(max_features0)
best_tree = DecisionTreeRegressor(max_depth=max_depth0, min_samples_split=min_samples_split0, min_samples_leaf=min_samples_leaf0, max_features=max_features0)
best_tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz

export_graphviz(best_tree, out_file='best_tree.dot', feature_names=X.columns)
print(open('best_tree.dot').read())
features = {'f' + str(i + 1):name for (i, name) in zip(range(len(df1.columns)), df1.columns)}
importances = best_tree.feature_importances_

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
               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);
from sklearn.ensemble import RandomForestRegressor
df = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
df.fit(X_train, y_train)
y_pred = df.predict(X_valid)

print(mean_squared_error(y_valid, y_pred))
print(df.score(X_valid, y_valid))
df_params_n_estimators = {'n_estimators': [100, 200, 350, 400, 500]}# количество деревьев n_estimators
df_grid_n_estimators = GridSearchCV(df, df_params_n_estimators, cv=kf, scoring='explained_variance', n_jobs = -1)
df_grid_n_estimators.fit(X_train, y_train)
n_estimators0 = list(df_grid_n_estimators.best_params_.values())[0]
n_estimators0
df = RandomForestRegressor(n_estimators = n_estimators0)#максимальная глубина дерева max_depth
df_params_max_depth = {'max_depth': np.arange(1, 20)}
df_grid_max_depth = GridSearchCV(df, df_params_max_depth, cv=kf, scoring='explained_variance', n_jobs = -1)
df_grid_max_depth.fit(X_train, y_train)
max_depth0 = list(df_grid_max_depth.best_params_.values())[0]
max_depth0
df = RandomForestRegressor(n_estimators = n_estimators0, max_depth = max_depth0, min_samples_split = min_samples_split0)
df_params_min_samples_leaf = {'min_samples_leaf': np.arange(1, 20)}#минимальное число объектов в листе min_samples_leaf
df_grid_min_samples_leaf = GridSearchCV(df, df_params_min_samples_leaf, cv=kf, scoring='explained_variance', n_jobs = -1)
df_grid_min_samples_leaf.fit(X_train, y_train)
min_samples_leaf0 = list(df_grid_min_samples_leaf.best_params_.values())[0]
min_samples_leaf0
df = RandomForestRegressor(n_estimators = n_estimators0, max_depth = max_depth0)#минимальное число объектов для разбиения во внутренней вершине min_samples_split
df_params_min_samples_split = {'min_samples_split': np.arange(1, 20)}
df_grid_min_samples_split = GridSearchCV(df, df_params_min_samples_split, cv=kf, scoring='explained_variance', n_jobs = -1)
df_grid_min_samples_split.fit(X_train, y_train)
min_samples_split0 = list(df_grid_min_samples_split.best_params_.values())[0]
min_samples_split0
df = RandomForestRegressor(n_estimators = n_estimators0, max_depth = max_depth0, min_samples_split = min_samples_split0, min_samples_leaf = min_samples_leaf0)
#максимальное количество признаков, рассматриваемых при поиске лучшего разбиения max_features.
df_params_max_features = {'max_features': np.arange(1, 20)}
df_grid_max_features = GridSearchCV(df, df_params_max_features, cv=kf, scoring='explained_variance', n_jobs = -1)
df_grid_max_features.fit(X_train, y_train)
max_features0 = list(df_grid_max_features.best_params_.values())[0]
max_features0
fig, ax = plt.subplots(nrows=3, ncols=2, sharey=False, figsize = (20,20))
#для каждого гиперпараметра по заданию-кривые
ax[0,0].set_xlabel("n_estimators")
ax[0,0].set_ylabel("mean_test_score")
ax[0,0].plot(df_params_n_estimators["n_estimators"], df_grid_n_estimators.cv_results_["mean_test_score"]);

ax[0,1].set_xlabel("max_depth")
ax[0,1].set_ylabel("mean_test_score")
ax[0,1].plot(df_params_max_depth['max_depth'], df_grid_max_depth.cv_results_["mean_test_score"]);

ax[1,0].set_xlabel("min_samples_split")
ax[1,0].set_ylabel("mean_test_score")
ax[1,0].plot(df_params_min_samples_split["min_samples_split"], df_grid_min_samples_split.cv_results_["mean_test_score"]);

ax[1,1].set_xlabel("min_samples_leaf")
ax[1,1].set_ylabel("mean_test_score")
ax[1,1].plot(df_params_min_samples_leaf["min_samples_leaf"],df_grid_min_samples_leaf.cv_results_["mean_test_score"]);

ax[2,0].set_xlabel("max_features")
ax[2,0].set_ylabel("mean_test_score")
ax[2,0].plot(df_params_max_features["max_features"], df_grid_max_features.cv_results_["mean_test_score"]);
plt.figure(figsize=(15,5))
plt.title("Feature importances")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);