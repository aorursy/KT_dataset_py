# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Загружаем данные

df = pd.read_csv('../input/cardio_train.csv', sep=';')

df.head()
# Предобработка

df['age_years'] = np.floor(df['age'] / 365.25)

new_df = pd.get_dummies(df, columns=['cholesterol', 'gluc']) # пункты 2, 3, 4

new_df.head()
from sklearn.model_selection import train_test_split

X = new_df.drop(['id', 'cardio', 'age'], axis=1)

y = new_df['cardio']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, random_state=2019) # random_state=2019
# Обучение дерева решений

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=3, random_state=2019)

tree.fit(X_train, y_train)
# Визуализация

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot', feature_names=X.columns)

print(open('tree.dot').read()) 

# Далее скопировать полученный текст на сайт https://dreampuf.github.io/GraphvizOnline/ и сгенерировать граф

# Вставить картинку в блокнот: ![](ссылка)
# Предсказания для валидационного множества

from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11),

               #'min_samples_leaf': np.arange(2, 11)

              }



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
tree_grid.best_estimator_
tree_grid.best_score_
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
# Отрисовка графиков

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True) # 2 графика рядом с одинаковым масштабом по оси Оу



ax[0].plot(tree_params['max_depth'], tree_grid.cv_results_['mean_test_score']) # accuracy vs max_depth

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('Mean accuracy on test set')



# ax[1].plot(tree_params['min_samples_leaf'], tree_grid.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

# ax[1].set_xlabel('min_samples_leaf')

# ax[1].set_ylabel('Mean accuracy on test set')
# Выбор и отрисовка наилучшего дерева



pd.DataFrame(tree_grid.cv_results_).head().T



best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)



# export_graphviz(best_tree, out_file='best_tree.dot')

# print(open('best_tree.dot').read()) 

# Далее скопировать полученный текст на сайт https://dreampuf.github.io/GraphvizOnline/ и сгенерировать граф

# Вставить картинку в блокнот: ![](ссылка)
from sklearn.ensemble import RandomForestClassifier

# Ваш код

rf = RandomForestClassifier(n_estimators=100, random_state=2019, max_depth=6)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
import matplotlib.pyplot as plt



# features = {'f'+str(i+1):name for (i, name) in zip(range(len(new_df.columns), df.columns))}



# Важность признаков



# from sklearn.ensemble import RandomForestClassifier

# forest = RandomForestClassifier(...)

# forest.fit(X-train, y_train)



# importances = forest.feature_importances_



# indices = np.argsort(importances)[::-1]

# # Plot the feature importancies of the forest

# num_to_plot = 10

# feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# # Print the feature ranking

# print("Feature ranking:")



# for f in range(num_to_plot):

#     print(f+1, features["f"+str(feature_indices[f])], importances[indices[f]])



# plt.figure(figsize=(15,5))

# plt.title("Feature importances")

# bars = plt.bar(range(num_to_plot), 

#                importances[indices[:num_to_plot]],

#                color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

#                align="center")

# ticks = plt.xticks(range(num_to_plot), 

#                    feature_indices)

# plt.xlim([-1, num_to_plot])

# plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);
# Ваш код
# Загрузим датасет и посмотрим, как выглядят данные

from sklearn.datasets import load_digits



data = load_digits()

X, y = data.data, data.target



X[0, :].reshape([8, 8])
# Нарисуем несколько картинок

fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 6))

for i in range(4):

    ax[i].imshow(X[i, :].reshape([8, 8]), cmap='Greys');
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2019)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



tree = DecisionTreeClassifier(max_depth=5, random_state=2019)

tree_pipe = Pipeline([('scaler', StandardScaler()), 

                      ('tree', DecisionTreeClassifier(max_depth=5, 

                                                      random_state=2019))]) # мы применили "конвейер"



tree.fit(X_train, y_train)

tree_pipe.fit(X_train, y_train)
tree_pred = tree.predict(X_valid)

print(accuracy_score(tree_pred, y_valid))
tree_pipe_pred = tree_pipe.predict(X_valid)

print(accuracy_score(tree_pipe_pred, y_valid))
tree_params = {'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],

               'max_features': [1, 2, 3, 5, 10, 20 ,30, 50, 64]}

knn_params = {'n_neighbors': np.arange(1, 51)}



# Ваш код