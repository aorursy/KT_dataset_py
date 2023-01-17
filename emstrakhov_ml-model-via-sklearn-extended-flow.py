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
df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', sep=';')

df.head()
# Возраст в днях --> Возраст в годах

df['age_years'] = np.floor(df['age'] / 365.25)



# Преобразование категориального признака gender путём кодирования с помощью map

df['gender'] = df['gender'].map({1:0, 2:1})



# get_dummies

df1 = pd.get_dummies(df, columns=['cholesterol', 'gluc'])



# drop'аем ненужные колонки id, age

df2 = df1.drop(['id', 'age'], axis=1)
df2.head()
# Импорт нужной функции

from sklearn.model_selection import train_test_split



# Создание X, y

# X --- вся таблица без таргета

# y --- таргет (целевая переменная)



X = df2.drop('cardio', axis=1) 

y = df2['cardio'] 



# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации

# random_state --- произвольное целое число, для воспроизводимости случайных результатов



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=10) # max_depth --- один из гиперпараметров дерева
tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
from sklearn.metrics import accuracy_score

print('Качество модели:', accuracy_score(y_valid, y_pred))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=12) # n_splits играет роль K

tree = DecisionTreeClassifier(max_depth=10)

scores = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))
from sklearn.model_selection import GridSearchCV



tree_params={'max_depth': np.arange(2, 15)} # словарь параметров (ключ: набор возможных значений)



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
# Смотрим лучшие значения параметров

print(tree_grid.best_params_)



# Лучшая модель

print(tree_grid.best_estimator_)
# Результаты кросс-валидации в виде таблицы

pd.DataFrame(tree_grid.cv_results_).T
# Рисуем валидационную кривую

# По оси х --- значения гиперпараметров (param_max_depth)

# По оси y --- значения метрики (mean_test_score)



import matplotlib.pyplot as plt

results_df = pd.DataFrame(tree_grid.cv_results_)

plt.plot(results_df['param_max_depth'], results_df['mean_test_score'])



# Подписываем оси и график

plt.xlabel('max_depth')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()