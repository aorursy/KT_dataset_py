# Убираем из блокнота лишние сообщения

import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()
iris_dataset.keys()
print(iris_dataset['DESCR'])
iris_dataset['target_names']
iris_dataset['feature_names']
type(iris_dataset['data'])
print(iris_dataset['data'].shape)

print(iris_dataset['target'].shape)
iris_dataset['data'][:5]
iris_dataset['target']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(iris_dataset['data'], 

                                                      iris_dataset['target'], 

                                                      test_size=0.3, 

                                                      random_state=19)
# Создаём dataframe из данных в массиве X_train

# Маркируем столбцы, используя строки в iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Создаём матрицу рассеяния из dataframe, цвет точек задаём с помощью y_train

grr = pd.plotting.scatter_matrix(iris_dataframe, 

                                 c=y_train, 

                                 figsize=(15, 15), 

                                 marker='o',

                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
# Создаём представителя класса модели, задаём необходимые гиперпараметры

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
# Обучаем модель на обучающей выборке

knn.fit(X_train, y_train)
# Строим предсказания на основе обученной модели

y_pred = knn.predict(X_valid)

y_pred
# Вычисляем метрику (меру) качества

knn.score(X_valid, y_valid)
# Другой способ для вычисления метрики

from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, iris_dataset['data'], iris_dataset['target'], 

                         cv=kf, scoring='accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 21)} # число соседей -- от 1 до 20

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=5) # или cv=kf

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
pd.DataFrame(knn_grid.cv_results_).T
# Предсказания на тестовой выборке для оптимального числа соседей

y_pred = knn_grid.predict(X_valid)

accuracy_score(y_valid, y_pred)
best_knn = KNeighborsClassifier(n_neighbors=15)

y_pred = best_knn.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)