



import numpy as np # linear algebra

import pandas as pd # data processing

# Загружаем данные

df = pd.read_csv('../input/cardio_train.csv', sep=';')

df.head()
# Предобработка

df['age'] = np.floor(df['age'] / 365.25)

df1=pd.get_dummies(df, columns=['cholesterol','gluc'])

df1.head()
from sklearn.model_selection import train_test_split

X = df1.drop(['id', 'cardio'], axis=1)

y = df1['cardio']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, random_state=2019) 
# Обучение дерева решений

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=3, random_state=2019)

tree.fit(X_train, y_train)
# Визуализация

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot')

print(open('tree.dot').read()) 

# Далее скопировать полученный текст на сайт https://dreampuf.github.io/GraphvizOnline/ и сгенерировать граф

# Вставить картинку в блокнот: ![](ссылка)
X_train.head()
# Предсказания для валидационного множества

from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
tree_params1 = {'min_samples_leaf': np.arange(2, 11)}



tree_grid1 = GridSearchCV(tree, tree_params1, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid1.fit(X_train, y_train)
# Отрисовка графиков

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True) # 2 графика рядом с одинаковым масштабом по оси Оу



ax[0].plot(tree_params['max_depth'], tree_grid.cv_results_['mean_test_score']) # accuracy vs max_depth

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('Mean accuracy on test set')



ax[1].plot(tree_params1['min_samples_leaf'], tree_grid1.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

ax[1].set_xlabel('min_samples_leaf')

ax[1].set_ylabel('Mean accuracy on test set')
# Выбор и отрисовка наилучшего дерева



pd.DataFrame(tree_grid.cv_results_).head().T



best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)



export_graphviz(best_tree, out_file='best_tree.dot')

print(open('best_tree.dot').read()) 

# Далее скопировать полученный текст на сайт https://dreampuf.github.io/GraphvizOnline/ и сгенерировать граф

# Вставить картинку в блокнот: ![](ссылка)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=2019, max_depth=6)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
# нет ограничения на глубину дерева

rf = RandomForestClassifier(n_estimators=100, random_state=2019)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
# кросс-валидация по гиперпараметру (количество деревьев в лесу)

rf_params_n_estimators = {'n_estimators': np.arange(100, 161, 10)}

rf_n_estimators = RandomForestClassifier(random_state=22)

rf_grid_n_estimators = GridSearchCV(rf_n_estimators, rf_params_n_estimators, cv=5, scoring='accuracy')

rf_grid_n_estimators.fit(X_train, y_train)



print(rf_grid_n_estimators.best_score_)

print(rf_grid_n_estimators.best_params_)

print(rf_grid_n_estimators.best_estimator_)
import matplotlib.pyplot as plt



fig, ax = plt.subplots()



ax.plot(rf_params_n_estimators['n_estimators'], rf_grid_n_estimators.cv_results_['mean_test_score']) 

ax.set_xlabel('n_estimators')

ax.set_ylabel('Mean accuracy on test set')
rf_params_max_depth = {'max_depth': np.arange(2, 11)}

rf_max_depth = RandomForestClassifier(n_estimators=160, random_state=22)

rf_grid_max_depth = GridSearchCV(rf_max_depth, rf_params_max_depth, cv=5, scoring='accuracy') 

rf_grid_max_depth.fit(X_train, y_train)



print(rf_grid_max_depth.best_score_)

print(rf_grid_max_depth.best_params_)

print(rf_grid_max_depth.best_estimator_)
fig, ax = plt.subplots() 



ax.plot(rf_params_max_depth['max_depth'], rf_grid_max_depth.cv_results_['mean_test_score']) # accuracy vs max_depth

ax.set_xlabel('max_depth')

ax.set_ylabel('Mean accuracy on test set')
feature = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature
# дерево после подбора гиперпараметров

feature = pd.Series(best_tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature
feature = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature