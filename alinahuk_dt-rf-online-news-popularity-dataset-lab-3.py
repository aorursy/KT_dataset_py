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
df.info()
df.describe().T
df['shares']
df['shares_log'] = np.log(df['shares'])

df1 = df.drop(['shares', 'url'], axis=1) # удалим shares

df1
X = df1.drop('shares_log', axis = 1)

y = df1['shares_log']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_new = scaler.fit_transform(X)

X_new
from sklearn.model_selection import train_test_split



# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации

# random_state --- произвольное целое число, для воспроизводимости случайных результатов

X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, test_size=0.25, random_state=42)
# Обучение дерева решений

from sklearn.tree import DecisionTreeRegressor 



tree = DecisionTreeRegressor(max_depth=3, random_state=42)

tree.fit(X_train, y_train)
# Предсказания для валидационного множества



y_pred = tree.predict(X_valid)

y_pred
from sklearn.metrics import mean_squared_error



mean_squared_error(y_valid, y_pred)
tree.score(X_valid, y_valid)
from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=True, random_state=42) 
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params_max_depth = {'max_depth': np.arange(3, 20)}

tree_grid_max_depth = GridSearchCV(tree, tree_params_max_depth, cv=kf, scoring='explained_variance', n_jobs = -1) # кросс-валидация по 5 блокам

tree_grid_max_depth.fit(X_train, y_train)
#чтобы получить наиболее правильное значение, будем записывать нужные нам значение в соответвующие переменные

max_depth_0 = list(tree_grid_max_depth.best_params_.values())[0]

print("Максимальная глубина: ", max_depth_0)

print("Наилучшая оценка качества модели: ", tree_grid_max_depth.best_score_)
tree_grid_max_depth.best_estimator_
tree = DecisionTreeRegressor(max_depth=max_depth_0)

tree_params_min_samples_split = {'min_samples_split': np.arange(3, 20)}

tree_grid_min_samples_split = GridSearchCV(tree, tree_params_min_samples_split, cv=kf, scoring='explained_variance', n_jobs = -1) # кросс-валидация по 5 блокам

tree_grid_min_samples_split.fit(X_train, y_train)
min_samples_split_0 = list(tree_grid_min_samples_split.best_params_.values())[0]

print("Минимальное число объектов для разбиения во внутренней вершине: ", min_samples_split_0)

print("Наилучшая оценка качества модели: ", tree_grid_min_samples_split.best_score_)
tree_grid_min_samples_split.best_estimator_
tree = DecisionTreeRegressor(max_depth=max_depth_0, min_samples_split=min_samples_split_0)

tree_params_min_samples_leaf = {'min_samples_leaf': np.arange(3, 20)}

tree_grid_min_samples_leaf = GridSearchCV(tree, tree_params_min_samples_leaf, cv=kf, scoring='explained_variance', n_jobs = -1) # кросс-валидация по 5 блокам

tree_grid_min_samples_leaf.fit(X_train, y_train)
min_samples_leaf_0 = list(tree_grid_min_samples_leaf.best_params_.values())[0]

print("Минимальное число объектов в листе: ", min_samples_leaf_0)

print("Наилучшая оценка качества модели: ", tree_grid_min_samples_leaf.best_score_)
tree_grid_min_samples_leaf.best_estimator_
tree = DecisionTreeRegressor(max_depth=max_depth_0, min_samples_split=min_samples_split_0, min_samples_leaf=min_samples_leaf_0)

tree_params_max_features = {'max_features': np.arange(3, 20)}

tree_grid_max_features = GridSearchCV(tree, tree_params_max_features, cv=kf, scoring='explained_variance', n_jobs = -1) # кросс-валидация по 5 блокам

tree_grid_max_features.fit(X_train, y_train)
max_features_0 = list(tree_grid_max_features.best_params_.values())[0]

print("Максимальное количество признаков, рассматриваемых при поиске лучшего разбиения: ", max_features_0)

print("Наилучшая оценка качества модели: ", tree_grid_max_features.best_score_)
tree_grid_max_features.best_estimator_
# Отрисовка графиков

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, figsize = (20,10))



ax[0, 0].plot(tree_params_max_depth['max_depth'], tree_grid_max_depth.cv_results_['mean_test_score']) # accuracy vs max_depth

ax[0, 0].set_xlabel('max_depth')

ax[0, 0].set_ylabel('Mean accuracy on test set')



ax[0, 1].plot(tree_params_min_samples_leaf['min_samples_leaf'], tree_grid_min_samples_leaf.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

ax[0, 1].set_xlabel('min_samples_leaf')

ax[0, 1].set_ylabel('Mean accuracy on test set')



ax[1, 0].plot(tree_params_min_samples_split['min_samples_split'], tree_grid_min_samples_split.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

ax[1, 0].set_xlabel('min_samples_split')

ax[1, 0].set_ylabel('Mean accuracy on test set')



ax[1, 1].plot(tree_params_max_features['max_features'], tree_grid_max_features.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

ax[1, 1].set_xlabel('max_features')

ax[1, 1].set_ylabel('Mean accuracy on test set')
print("max_depth =", max_depth_0)

print("min_samples_split =", min_samples_split_0)

print("min_samples_leaf =", min_samples_leaf_0)

print("max_features =", max_features_0)
best_tree = DecisionTreeRegressor(max_depth=max_depth_0, min_samples_split=min_samples_split_0, min_samples_leaf=min_samples_leaf_0, max_features=max_features_0)

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

# Ваш код

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



mean_squared_error(y_valid, y_pred)
rf.score(X_valid, y_valid)
rf_params_n_estimators = {'n_estimators': [100, 200, 350, 400, 500]}

rf_grid_n_estimators = GridSearchCV(rf, rf_params_n_estimators, cv=kf, scoring='explained_variance', n_jobs = -1)

rf_grid_n_estimators.fit(X_train, y_train)
n_estimators_0 = list(rf_grid_n_estimators.best_params_.values())[0]

n_estimators_0
rf = RandomForestRegressor(n_estimators = n_estimators_0)

rf_params_max_depth = {'max_depth': np.arange(1, 20)}

rf_grid_max_depth = GridSearchCV(rf, rf_params_max_depth, cv=kf, scoring='explained_variance', n_jobs = -1)

rf_grid_max_depth.fit(X_train, y_train)
max_depth_0 = list(rf_grid_max_depth.best_params_.values())[0]

max_depth_0
rf = RandomForestRegressor(n_estimators = n_estimators_0, max_depth = max_depth_0)

rf_params_min_samples_split = {'min_samples_split': np.arange(1, 20)}

rf_grid_min_samples_split = GridSearchCV(rf, rf_params_min_samples_split, cv=kf, scoring='explained_variance', n_jobs = -1)

rf_grid_min_samples_split.fit(X_train, y_train)
min_samples_split_0 = list(rf_grid_min_samples_split.best_params_.values())[0]

min_samples_split_0
rf = RandomForestRegressor(n_estimators = n_estimators_0, max_depth = max_depth_0, min_samples_split = min_samples_split_0)

rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(1, 20)}

rf_grid_min_samples_leaf = GridSearchCV(rf, rf_params_min_samples_leaf, cv=kf, scoring='explained_variance', n_jobs = -1)

rf_grid_min_samples_leaf.fit(X_train, y_train)
min_samples_leaf_0 = list(rf_grid_min_samples_leaf.best_params_.values())[0]

min_samples_leaf_0
rf = RandomForestRegressor(n_estimators = n_estimators_0, max_depth = max_depth_0, min_samples_split = min_samples_split_0, min_samples_leaf = min_samples_leaf_0)

rf_params_max_features = {'max_features': np.arange(1, 20)}

rf_grid_max_features = GridSearchCV(rf, rf_params_max_features, cv=kf, scoring='explained_variance', n_jobs = -1)

rf_grid_max_features.fit(X_train, y_train)
max_features_0 = list(rf_grid_max_features.best_params_.values())[0]

max_features_0
print("n_estimators =", n_estimators_0)

print("max_depth =", max_depth_0)

print("min_samples_split =", min_samples_split_0)

print("min_samples_leaf =", min_samples_leaf_0)

print("max_features =", max_features_0)
fig, ax = plt.subplots(nrows=3, ncols=2, sharey=False, figsize = (20,20))



ax[0,0].set_xlabel("n_estimators")

ax[0,0].set_ylabel("mean_test_score")

ax[0,0].plot(rf_params_n_estimators["n_estimators"], rf_grid_n_estimators.cv_results_["mean_test_score"]);



ax[0,1].set_xlabel("max_depth")

ax[0,1].set_ylabel("mean_test_score")

ax[0,1].plot(rf_params_max_depth['max_depth'], rf_grid_max_depth.cv_results_["mean_test_score"]);



ax[1,0].set_xlabel("min_samples_split")

ax[1,0].set_ylabel("mean_test_score")

ax[1,0].plot(rf_params_min_samples_split["min_samples_split"], rf_grid_min_samples_split.cv_results_["mean_test_score"]);



ax[1,1].set_xlabel("min_samples_leaf")

ax[1,1].set_ylabel("mean_test_score")

ax[1,1].plot(rf_params_min_samples_leaf["min_samples_leaf"],rf_grid_min_samples_leaf.cv_results_["mean_test_score"]);



ax[2,0].set_xlabel("max_features")

ax[2,0].set_ylabel("mean_test_score")

ax[2,0].plot(rf_params_max_features["max_features"], rf_grid_max_features.cv_results_["mean_test_score"]);
import matplotlib.pyplot as plt



features = {'f'+str(i+1):name for (i, name) in zip(range(len(df1.columns)), df1.columns)}



# Важность признаков



from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=200, max_depth=9, random_state=42)

forest.fit(X_train, y_train)



importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features["f"+str(feature_indices[f])], importances[indices[f]])
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