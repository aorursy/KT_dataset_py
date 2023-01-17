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
import numpy as np 
import pandas as pd
df = pd.read_csv('../input/vodafone-subset-3.csv')
dataset = df[['target','ROUM','AVG_ARPU','car','gender','ecommerce_score','gas_stations_sms','phone_value','calls_duration_in_weekdays','calls_duration_out_weekdays','calls_count_in_weekends','calls_count_out_weekends']]
dataset
dataset.shape
data = dataset.drop('target',axis = 1)#датасет без 'target'
data
data.shape #убедились
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(data, 
                                                      dataset['target'], 
                                                      test_size=0.25, 
                                                      random_state=123)
dataset.keys()
dataset['target']
dataset['ecommerce_score']
type(dataset['ecommerce_score'])
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
X_train.shape
y_train.shape
knn.fit(X_train, y_train)#Обучаем модель на обучающей выборке
y_pred = knn.predict(X_valid)#Строим предсказания на основе обученной модели
y_pred
knn.score(X_valid, y_valid)# Вычисляем метрику (меру) качества
from sklearn.metrics import accuracy_score# Другой способ для вычисления метрики
accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, dataset, dataset['target'], cv=kf, scoring = 'accuracy')
print(scores)
mean_score = scores.mean()
print(mean_score)
import matplotlib
from sklearn.model_selection import GridSearchCV
knn_params = {'n_neighbors': np.arange(1, 51)} # число соседей -- от 1 до 50
knn_grid = GridSearchCV(knn, 
                        knn_params, 
                        scoring='accuracy',
                        cv=5) # или cv=kf
knn_grid.fit(X_train, y_train)

knn_grid.best_estimator_
knn_grid.best_score_
pd.DataFrame(knn_grid.cv_results_).T
# Предсказания на тестовой выборке для оптимального числа соседей
y_pred = knn_grid.predict(X_valid)
accuracy_score(y_valid, y_pred)
best_knn = KNeighborsClassifier(n_neighbors=15)
y_pred = best_knn.fit(X_train, y_train).predict(X_valid)
accuracy_score(y_valid, y_pred)
#Построим график значений метрики в зависимости от k 
a = []
score = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = i )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_valid)
    a.append(i)
    score.append(accuracy_score(y_valid, y_pred))
matplotlib.pyplot.plot(a, score)
knn_grid.best_params_
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state=2019) #максимальная глубина дерева = 3
tree.fit(X_train, y_train)
#Визуализация 
#from sklearn.tree import export_graphviz

#export_graphviz(tree, out_file='tree.dot')
#print(open('tree.dot').read()) 
# Предсказания для валидационного множества
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
# Кросс-валидация и подбор гиперпараметров
from sklearn.model_selection import GridSearchCV

tree_params_a= {'max_depth': np.arange(2, 11)}

tree_grid_a = GridSearchCV(tree, tree_params_a, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid_a.fit(X_train, y_train)
# Кросс-валидация и подбор гиперпараметров
from sklearn.model_selection import GridSearchCV

tree_params_b= {'min_samples_leaf': np.arange(2, 11)}

tree_grid_b = GridSearchCV(tree, tree_params_b, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid_b.fit(X_train, y_train)
# Отрисовка графиков
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True) # 2 графика рядом с одинаковым масштабом по оси Оу

ax[0].plot(tree_params_a['max_depth'], tree_grid_a.cv_results_['mean_test_score']) # accuracy vs max_depth
ax[0].set_xlabel('max_depth')
ax[0].set_ylabel('Mean accuracy on test set')

ax[1].plot(tree_params_b['min_samples_leaf'], tree_grid_b.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf
ax[1].set_xlabel('min_samples_leaf')
ax[1].set_ylabel('Mean accuracy on test set')
# Кросс-валидация и подбор гиперпараметров
from sklearn.model_selection import GridSearchCV

tree_params = {'max_depth': np.arange(2, 11),
               'min_samples_leaf': np.arange(2, 11)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам
tree_grid.fit(X_train, y_train)
# Выбор наилучшего дерева

pd.DataFrame(tree_grid.cv_results_).head().T

best_tree = tree_grid.best_estimator_
y_pred = best_tree.predict(X_valid)
accuracy_score(y_valid, y_pred)
#export_graphviz(best_tree, out_file='best_tree.dot')
#print(open('best_tree.dot').read()) 
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision:', precision_score(y_valid, y_pred,average='macro'))
print('Recall:', recall_score(y_valid, y_pred,average='macro'))
print('F1 score:', f1_score(y_valid, y_pred,average='macro'))