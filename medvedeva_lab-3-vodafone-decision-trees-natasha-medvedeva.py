# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

vodafone_subset_6 = pd.read_csv("../input/vodafone6nm/vodafone-subset-6.csv")
vodafone_subset_6.head(10)
df = vodafone_subset_6[['target', 'ROUM', 'phone_value', 'DATA_VOLUME_WEEKDAYS', 'DATA_VOLUME_WEEKENDS', 
                        'device_brand', 'software_os_vendor', 'software_os_name', 'software_os_version', 'device_type_rus',
                       'AVG_ARPU', 'lifetime', 'how_long_same_model', 'ecommerce_score',
                        'banks_sms_count', 'instagram_volume', 'viber_volume', 'linkedin_volume', 'tinder_volume', 'telegram_volume', 'google_volume', 'whatsapp_volume', 'youtube_volume']]
df.head()
df_1 = pd.get_dummies(df, columns=['phone_value', 'device_brand', 'software_os_vendor', 'software_os_name', 'software_os_version', 'device_type_rus'])
df_1.head()
X = df_1.drop('target', axis=1)
y = df_1['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=22)
tree = DecisionTreeClassifier(max_depth=3, random_state=22)
tree.fit(X_train, y_train)
export_graphviz(tree, out_file='tree.dot')
print(open('tree.dot').read()) 
y_pred = tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
tree_params = {'max_depth': np.arange(2, 11),
               'min_samples_leaf': np.arange(2, 11)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
print(tree_grid.best_score_)
print(tree_grid.best_params_)
print(tree_grid.best_estimator_)
best_tree = tree_grid.best_estimator_
y_pred = best_tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
pd.DataFrame(tree_grid.cv_results_).T
tree_params_max_depth = {'max_depth': np.arange(2, 11)}

tree_grid_max_depth = GridSearchCV(tree, tree_params_max_depth, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid_max_depth.fit(X_train, y_train)
print(tree_grid_max_depth.best_score_)
print(tree_grid_max_depth.best_params_)
print(tree_grid_max_depth.best_estimator_)
pd.DataFrame(tree_grid_max_depth.cv_results_).T
tree_params_min_samples_leaf = {'min_samples_leaf': np.arange(2, 11)}

tree_grid_min_samples_leaf = GridSearchCV(tree, tree_params_min_samples_leaf, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid_min_samples_leaf.fit(X_train, y_train)
print(tree_grid_min_samples_leaf.best_score_)
print(tree_grid_min_samples_leaf.best_params_)
print(tree_grid_min_samples_leaf.best_estimator_)
pd.DataFrame(tree_grid_min_samples_leaf.cv_results_).T
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 4)) # 2 графика рядом с одинаковым масштабом по оси Оу

ax[0].plot(tree_params_max_depth['max_depth'], tree_grid_max_depth.cv_results_['mean_test_score']) # accuracy vs max_depth
ax[0].set_xlabel('max_depth')
ax[0].set_ylabel('Mean accuracy on test set')

ax[1].plot(tree_params_min_samples_leaf['min_samples_leaf'], tree_grid_min_samples_leaf.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf
ax[1].set_xlabel('min_samples_leaf')
ax[1].set_ylabel('Mean accuracy on test set')

plt.show()
# best_tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=6, random_state=22)
# best_tree.fit(X_train, y_train)

# y_pred = best_tree.predict(X_valid)
# print(accuracy_score(y_valid, y_pred))
best_tree = tree_grid.best_estimator_
y_pred = best_tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
best_tree_max_depth = tree_grid_max_depth.best_estimator_
y_pred = best_tree_max_depth.predict(X_valid)
accuracy_score(y_valid, y_pred)
export_graphviz(best_tree_max_depth, out_file='tree.dot')
print(open('tree.dot').read()) 
rf = RandomForestClassifier(n_estimators=100, random_state=22)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)

accuracy_score(y_valid, y_pred)
rf = RandomForestClassifier(n_estimators=100, random_state=22, max_depth=4)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)

accuracy_score(y_valid, y_pred)
rf_params_n_estimators = {'n_estimators': np.arange(10, 201, 10)}
rf_n_estimators = RandomForestClassifier(random_state=22)
rf_grid_n_estimators = GridSearchCV(rf_n_estimators, rf_params_n_estimators, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
rf_grid_n_estimators.fit(X_train, y_train)

print(rf_grid_n_estimators.best_score_)
print(rf_grid_n_estimators.best_params_)
print(rf_grid_n_estimators.best_estimator_)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(rf_params_n_estimators['n_estimators'], rf_grid_n_estimators.cv_results_['mean_test_score']) # accuracy vs max_depth
ax.set_xlabel('n_estimators')
ax.set_ylabel('Mean accuracy on test set')
rf_params_n_estimators = {'n_estimators': np.arange(201, 301, 10)}
rf_n_estimators = RandomForestClassifier(random_state=22)
rf_grid_n_estimators = GridSearchCV(rf_n_estimators, rf_params_n_estimators, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
rf_grid_n_estimators.fit(X_train, y_train)

print(rf_grid_n_estimators.best_score_)
print(rf_grid_n_estimators.best_params_)
print(rf_grid_n_estimators.best_estimator_)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(rf_params_n_estimators['n_estimators'], rf_grid_n_estimators.cv_results_['mean_test_score']) # accuracy vs max_depth
ax.set_xlabel('n_estimators')
ax.set_ylabel('Mean accuracy on test set')
rf_params_n_estimators = {'n_estimators': np.arange(171, 190)}
rf_n_estimators = RandomForestClassifier(random_state=22)
rf_grid_n_estimators = GridSearchCV(rf_n_estimators, rf_params_n_estimators, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
rf_grid_n_estimators.fit(X_train, y_train)

print(rf_grid_n_estimators.best_score_)
print(rf_grid_n_estimators.best_params_)
print(rf_grid_n_estimators.best_estimator_)
rf_params_max_features = {'max_features': np.arange(5, 277, 5)}
rf_max_features = RandomForestClassifier(n_estimators=183, random_state=22)
rf_grid_max_features = GridSearchCV(rf_max_features, rf_params_max_features, cv=5, scoring='accuracy')
rf_grid_max_features.fit(X_train, y_train)

print(rf_grid_max_features.best_score_)
print(rf_grid_max_features.best_params_)
print(rf_grid_max_features.best_estimator_)
fig, ax = plt.subplots() 

ax.plot(rf_params_max_features['max_features'], rf_grid_max_features.cv_results_['mean_test_score']) # accuracy vs max_depth
ax.set_xlabel('max_features')
ax.set_ylabel('Mean accuracy on test set')
rf_params_max_features = {'max_features': np.arange(126, 135)}
rf_max_features = RandomForestClassifier(n_estimators=183, random_state=22)
rf_grid_max_features = GridSearchCV(rf_max_features, rf_params_max_features, cv=5, scoring='accuracy')
rf_grid_max_features.fit(X_train, y_train)

print(rf_grid_max_features.best_score_)
print(rf_grid_max_features.best_params_)
print(rf_grid_max_features.best_estimator_)
rf_params_max_depth = {'max_depth': np.arange(2, 11)}
rf_max_depth = RandomForestClassifier(n_estimators=183, max_features=130, random_state=22)
rf_grid_max_depth = GridSearchCV(rf_max_depth, rf_params_max_depth, cv=5, scoring='accuracy') 
rf_grid_max_depth.fit(X_train, y_train)

print(rf_grid_max_depth.best_score_)
print(rf_grid_max_depth.best_params_)
print(rf_grid_max_depth.best_estimator_)
fig, ax = plt.subplots() 

ax.plot(rf_params_max_depth['max_depth'], rf_grid_max_depth.cv_results_['mean_test_score']) # accuracy vs max_depth
ax.set_xlabel('max_depth')
ax.set_ylabel('Mean accuracy on test set')
rf_params_max_depth = {'max_depth': np.arange(11, 21)}
rf_max_depth = RandomForestClassifier(n_estimators=183, max_features=130, random_state=22)
rf_grid_max_depth = GridSearchCV(rf_max_depth, rf_params_max_depth, cv=5, scoring='accuracy') 
rf_grid_max_depth.fit(X_train, y_train)

print(rf_grid_max_depth.best_score_)
print(rf_grid_max_depth.best_params_)
print(rf_grid_max_depth.best_estimator_)
rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(3, 10, 2)}
rf_min_samples_leaf = RandomForestClassifier(n_estimators=183, max_features=130, max_depth=8, random_state=22)
rf_grid_min_samples_leaf = GridSearchCV(rf_min_samples_leaf, rf_params_min_samples_leaf, cv=5, scoring='accuracy')
rf_grid_min_samples_leaf.fit(X_train, y_train)

print(rf_grid_min_samples_leaf.best_score_)
print(rf_grid_min_samples_leaf.best_params_)
print(rf_grid_min_samples_leaf.best_estimator_)
fig, ax = plt.subplots() 

ax.plot(rf_params_min_samples_leaf['min_samples_leaf'], rf_grid_min_samples_leaf.cv_results_['mean_test_score']) # accuracy vs max_depth
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('Mean accuracy on test set')
rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(3, 21)}
rf_min_samples_leaf = RandomForestClassifier(n_estimators=183, max_features=130, max_depth=8, random_state=22)
rf_grid_min_samples_leaf = GridSearchCV(rf_min_samples_leaf, rf_params_min_samples_leaf, cv=5, scoring='accuracy')
rf_grid_min_samples_leaf.fit(X_train, y_train)

print(rf_grid_min_samples_leaf.best_score_)
print(rf_grid_min_samples_leaf.best_params_)
print(rf_grid_min_samples_leaf.best_estimator_)
fig, ax = plt.subplots() 

ax.plot(rf_params_min_samples_leaf['min_samples_leaf'], rf_grid_min_samples_leaf.cv_results_['mean_test_score']) # accuracy vs max_depth
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('Mean accuracy on test set')
# rf_params = {'n_estimators': np.arange(170, 196, 5), 'max_features': np.arange(70, 277, 20), 'max_depth': np.arange(7, 13)}
# rf = RandomForestClassifier(min_samples_leaf=8, random_state=22)
# rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
# rf_grid.fit(X_train, y_train)

# print(rf_grid.best_score_)
# print(rf_grid.best_params_)
# print(rf_grid.best_estimator_)
# import matplotlib.pyplot as plt

# features = dict(zip(range(len(X.columns)), X.columns))

# # Важность признаков
# importances = rf_grid.best_estimator_.feature_importances_

# indices = np.argsort(importances)[::-1]
# # Plot the feature importancies of the forest
# num_to_plot = max(10, len(X.columns))
# feature_indices = [ind for ind in indices[:num_to_plot]]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(num_to_plot):
#     print(f+1, features[feature_indices[f]], importances[indices[f]])

# plt.figure(figsize=(15,5))
# plt.title("Feature importances")
# bars = plt.bar(range(num_to_plot), 
#                importances[indices[:num_to_plot]],
#                color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
#                align="center")
# ticks = plt.xticks(range(num_to_plot), 
#                    feature_indices)
# plt.xlim([-1, num_to_plot])
# plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);



# import matplotlib.pyplot as plt

# features = dict(zip(range(len(X.columns)), X.columns))

# # Важность признаков
# importances = rf_grid.best_estimator_.feature_importances_

# indices = np.argsort(importances)[::-1]
# # Plot the feature importancies of the forest
# num_to_plot = min(10, len(X.columns))
# feature_indices = [ind for ind in indices[:num_to_plot]]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(num_to_plot):
#     print(f+1, features[feature_indices[f]], importances[indices[f]])

# plt.figure(figsize=(15,5))
# plt.title("Feature importances")
# bars = plt.bar(range(num_to_plot), 
#                importances[indices[:num_to_plot]],
#                color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
#                align="center")
# ticks = plt.xticks(range(num_to_plot), 
#                    feature_indices)
# plt.xlim([-1, num_to_plot])
# plt.legend(bars, [u''.join(features[i]) for i in feature_indices])
# plt.show()