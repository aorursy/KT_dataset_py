# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#Загрузка данных

df = pd.read_csv("/kaggle/input/bank-marketing-dataset/bank.csv")

df.head()
# Статистика по числовым признакам

df.describe().T
df.info()
df['deposit']
new_values = {'yes':  1, 'no': 0} 

df['deposit1'] = df['deposit'].map(new_values)

df['housing1'] = df['housing'].map(new_values)

df['loan1'] = df['loan'].map(new_values)



df['default1'] = df['default'].map(new_values)

df['contact'].value_counts()



new_values_1 = {'cellular':  1, 'unknown': 0,'telephone' : 1 } 

df['contact1'] = df['contact'].map(new_values_1)

df['contact1'].value_counts()



new_values_2 = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec':12}

df['month1'] = df['month'].map(new_values_2)



df = df.drop(['deposit','housing','loan','default','contact', 'month'], axis = 1)



df = pd.get_dummies(df, columns=['marital','poutcome','education'])

df.head().T
from sklearn.model_selection import train_test_split



X = df.drop('deposit1', axis=1).drop('job',axis = 1)

y = df['deposit1'] 





# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации

# random_state --- произвольное целое число, для воспроизводимости случайных результатов



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)

X.info()
print(X_train.shape, y_train.shape)
print( X_valid.shape, y_valid.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



tree = DecisionTreeClassifier(max_depth=3, random_state=2019)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)

print('Качество модели:', accuracy_score(y_pred, y_valid))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42) # n_splits играет роль K

scores = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')

print('Массив значений метрики:', scores)
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params_depth = {'max_depth': np.arange(2, 11)}



tree_grid_depth = GridSearchCV(tree, tree_params_depth, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid_depth.fit(X_train, y_train)

tree_grid_depth.best_params_

m_depth = tree_grid_depth.best_params_['max_depth']

print(m_depth)
tree_params_split = {'min_samples_split': np.arange(2,21)}



tree = DecisionTreeClassifier(max_depth = m_depth)

tree_grid_samples_split = GridSearchCV(tree, tree_params_split, cv=5, scoring='accuracy')

tree_grid_samples_split.fit(X_train, y_train)

tree_grid_samples_split.best_params_

m_split = tree_grid_samples_split.best_params_['min_samples_split']

print(m_split)
tree_params_leaf = {'min_samples_leaf':np.arange(2,21)}



tree = DecisionTreeClassifier(max_depth = m_depth, min_samples_split = m_split)

tree_grid_samples_leaf = GridSearchCV(tree, tree_params_leaf, cv=5, scoring='accuracy')

tree_grid_samples_leaf.fit(X_train, y_train)

tree_grid_samples_leaf.best_params_

m_leaf = tree_grid_samples_leaf.best_params_['min_samples_leaf']

print(m_leaf)
tree_params_features = {'max_features':np.arange(2,21)}



tree = DecisionTreeClassifier(max_depth = m_depth, min_samples_split = m_split, min_samples_leaf = m_leaf)

tree_grid_features = GridSearchCV(tree, tree_params_features, cv=5, scoring='accuracy')

tree_grid_features.fit(X_train, y_train)

tree_grid_features.best_params_

m_features = tree_grid_features.best_params_['max_features']

print(m_features)
# Валидационная кривая

import matplotlib.pyplot as plt



fig, ax = plt.subplots(2, 2, figsize = (10,10))



ax[0, 0].set_xlabel('max_depth')

ax[0, 0].set_ylabel('accuracy score')

ax[0, 0].plot(tree_params_depth['max_depth'], tree_grid_depth.cv_results_['mean_test_score']);



ax[0, 1].set_xlabel('min_samples_split')

ax[0, 1].set_ylabel('accuracy score')

ax[0, 1].plot(tree_params_split['min_samples_split'], tree_grid_samples_split.cv_results_['mean_test_score']);



ax[1, 0].set_xlabel('min_samples_leaf')

ax[1, 0].set_ylabel('accuracy score')

ax[1, 0].plot(tree_params_leaf['min_samples_leaf'], tree_grid_samples_leaf.cv_results_['mean_test_score']);



ax[1, 1].set_xlabel('max_features')

ax[1, 1].set_ylabel('accuracy score')

ax[1, 1].plot(tree_params_features['max_features'], tree_grid_features.cv_results_['mean_test_score']);

tree = DecisionTreeClassifier(max_depth = m_depth, min_samples_split = m_split, min_samples_leaf = m_leaf, max_features = m_features, random_state=19)

tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz



export_graphviz(tree, out_file = 'tree.dot', feature_names = X.columns)

print(open('tree.dot').read())
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
# Обучение случайного леса

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, random_state=2019)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_valid)



from sklearn.metrics import accuracy_score



print(accuracy_score(y_valid, y_pred))
rf_params_estimators = {"n_estimators":[50, 100, 150, 200, 350, 400]}

rf_estimators = GridSearchCV(rf, rf_params_estimators, cv=5, scoring='accuracy')

rf_estimators.fit(X_train, y_train)

rf_estimators.best_params_

rf_params_depth = {'max_depth': np.arange(2, 11)}

rf = RandomForestClassifier(random_state = 19, n_estimators = 200)

rf_grid_depth = GridSearchCV(rf, rf_params_depth, cv=5, scoring='accuracy') 

rf_grid_depth.fit(X_train, y_train)

rf_grid_depth.best_params_
rf_params_split = {'min_samples_split': np.arange(2, 21)}

rf = RandomForestClassifier(random_state = 19, n_estimators = 200, max_depth = 10)

rf_grid_split = GridSearchCV(rf, rf_params_split, cv=5, scoring='accuracy') 

rf_grid_split.fit(X_train, y_train)

rf_grid_split.best_params_
rf_params_leaf = {'min_samples_leaf': np.arange(2, 21)}

rf = RandomForestClassifier(random_state = 19, n_estimators = 200, max_depth = 10, min_samples_split = 5)

rf_grid_leaf = GridSearchCV(rf, rf_params_leaf, cv=5, scoring='accuracy') 

rf_grid_leaf.fit(X_train, y_train)

rf_grid_leaf.best_params_
rf_params_features = {'max_features': np.arange(2, 21)}

rf = RandomForestClassifier(random_state = 19, n_estimators = 200, max_depth = 10, min_samples_split = 5, min_samples_leaf= 7)

rf_grid_features = GridSearchCV(rf, rf_params_features, cv=5, scoring='accuracy') 

rf_grid_features.fit(X_train, y_train)

rf_grid_features.best_params_
# Валидационная кривая

import matplotlib.pyplot as plt



fig, ax = plt.subplots(2, 2, figsize = (10,10))



ax[0,0].set_xlabel('n_estimators')

ax[0,0].set_ylabel('accuracy score')

ax[0,0].plot(rf_params_estimators['n_estimators'], rf_estimators.cv_results_["mean_test_score"]);



ax[0, 1].set_xlabel('max_depth')

ax[0, 1].set_ylabel('accuracy score')

ax[0, 1].plot(rf_params_depth['max_depth'], rf_grid_depth.cv_results_['mean_test_score']);



ax[1, 0].set_xlabel('min_samples_split')

ax[1, 0].set_ylabel('accuracy score')

ax[1, 0].plot(rf_params_split['min_samples_split'], rf_grid_split.cv_results_['mean_test_score']);



ax[1, 1].set_xlabel('min_samples_leaf')

ax[1, 1].set_ylabel('accuracy score')

ax[1, 1].plot(rf_params_leaf['min_samples_leaf'], rf_grid_leaf.cv_results_['mean_test_score']);



ax[1, 2].set_xlabel('max_features')

ax[1, 2].set_ylabel('accuracy score')

ax[1, 2].plot(rf_params_features['max_features'], rf_grid_features.cv_results_['mean_test_score']);
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);