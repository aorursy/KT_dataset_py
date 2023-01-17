# Цель прогнозируемой модели: по данным из датасета спрогнозировать популярность публикации. 

# Целевая пременная: shares - количественная оценка, отражающая, сколько раз поделились ползователи публикацией.

# Датасет содержит информацию о публикациях на сайтах (новости).

# Данная задача - это задача регресии.Прогноз на основе выборки объектов с различными признаками. На выходе должно получиться вещественное число.

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Считываем данные.

df = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv', sep=',')

df.head()

# Спасибо за подсказку
# drop'аем (удаляем) ненужную колонку "url". 

df1 = df.drop(['url'], axis=1)

df1.head()
df.describe().T
# Извлекаем целевой признак (target).

target = df['shares']

target
# Распределение значений target-переменной.

import seaborn as sns

snsplot = sns.kdeplot(df['shares'], shade=True)

fig = snsplot.get_figure()
# Проверка на нормальное распредление.

from scipy.stats import normaltest

import seaborn as sns

import matplotlib.pyplot as plt

df['shares_log'] = np.log(df['shares']) 

stat, p = normaltest(df['shares_log'])

print('p-value:', p)

snsplot = sns.kdeplot(df['shares_log'], shade=True)

fig = snsplot.get_figure()

# В нашем случае график напоминает нормальное распредление, но т.к. значение p-value меньше 0,05, то данное распределение нормальным не является.
# Масштабирование данных.

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

StandardScaler(df1)
# Импорт нужной функции.

from sklearn.model_selection import train_test_split



# Создание X, y.

# X --- вся таблица без target.

# y --- target (целевая переменная).



X = df1.drop('shares', axis=1) 

y = df1['shares'] 



# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации.

# random_state --- произвольное целое число, для воспроизводимости случайных результатов.



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)

X.head()
#  Масштабирование данных. Нам поможет в этом класс StandardScaler.

X_new = scaler.fit_transform(X)

X_new
# Обучаем алгоритм классификации KNeighborsRegressor.

from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors = 10)

neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_valid)

y_pred



# Оценка модели валидационной выборки с помошью mean_squared_error.

from sklearn.metrics import mean_squared_error

print("Оценка модели валидационной выборки ",mean_squared_error(y_valid, y_pred))

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



KFOLD = KFold(n_splits=5, shuffle=True, random_state=12) 

# Создаем генератор разбиений, который перемешивает выборку перед создани-

#ем блоков ( shuffle=True ). Число блоков n_splits равно 5. Задаем также

#параметр random_state для воспроизводимости результатов. В нашем случае = 12.

tree = DecisionTreeClassifier(max_depth=10)

scores = cross_val_score(tree, X, y, cv=KFOLD, scoring='accuracy')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))


# Осуществление кросс-валидации модели при числе соседей k ∈ [1;50].

from sklearn.model_selection import GridSearchCV

neigh_params = {'n_neighbors': np.arange(1, 51)}

neigh_grid = GridSearchCV(neigh, 

                        neigh_params, 

                        scoring='explained_variance',

                        cv = KFOLD)

neigh_grid.fit(X_train, y_train)

# Смотрим лучшие значения параметров

print("Лучшее значение параметров:", neigh_grid.best_params_)



# Лучшая модель

print("Лучшая модель: ",neigh_grid.best_score_)
# Результаты кросс-валидации в виде таблицы

pd.DataFrame(neigh_grid.cv_results_).T

# Рисуем валидационную кривую

# По оси х --- значения гиперпараметров (param_n_neighbors)

# По оси y --- значения метрики (mean_test_score)

import matplotlib.pyplot as plt

results_df = pd.DataFrame(neigh_grid.cv_results_)

plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])

# Подписываем оси и график

plt.xlabel('param_n_neighbors')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
# Воспользуемся метрикой Минковского.Переберем разные варианты значений параметра p по сетке

#от 1 до 10 с таким шагом, чтобы всего было протестировано 200 вариантов 

#(удобно использовать функцию numpy.linspace ).Воспользуемся KNeighborsRegressor с оптимальным 

#значением n_neighbors , найденным ранее. Зададим опцию weights='distance' –– данный параметр 

#добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.



from sklearn.neighbors import KNeighborsRegressor as neigh

KFOLD=KFold(n_splits = 5, shuffle = True, random_state = 42) 

maximum  = -1

max_p = 0

#for p in param_p:

for p in np.linspace(1, 10, 2):

    kNN = KNeighborsRegressor(n_neighbors = 44, p = p, weights='distance', metric='minkowski')

    kNN.fit(X_train, y_train)

    y_pred_new = kNN.predict(X_valid)

    scores = cross_val_score(kNN, X, y, cv = KFOLD, scoring = 'neg_mean_squared_error')

    new_score = scores.mean()

    print("P-value:", p, "; Score:", new_score)

    if new_score > maximum:

        maximum = new_score

        max_p = p

# Остановил операцию, т.к. она заберет бОльшую часть моей жизни:)

#Изменил условия: поменял 200 вариантов на 2: np.linspace(1, 10, 2)
# Воспользуемся метрическим методом RadiusNeighborsRegressor для задачи регрессии.

from sklearn.neighbors import RadiusNeighborsRegressor

radneigh = RadiusNeighborsRegressor(radius = 5)

radneigh.fit(X_train, y_train)

y_pred = radneigh.predict(X_valid)

y_pred