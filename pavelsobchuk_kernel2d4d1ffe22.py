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


# Считываем данные

df = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')



# Назначаем имена колонок

columns = ('meanIntProf standardIntProf excessIntProf skewnessIntProf meanDMsnr standardDMsnr excassDMsnr skewnessDMsnr class')





df.columns = columns.split() #этот метод разделит датасет по колонкам как в массиве columns
df.head()
# Проверим наши данные. Видим, что нет пропущенных значений и нет необходимости в перекодировке данных, 

# т.к. категориальные данные отсутствуют

df.info()
#Убедимся в отсутствии категориальных признаков

len(list(df.columns)) - len(df._get_numeric_data().columns)
# посмотрим инфо по выборке

df.describe()
from matplotlib import pyplot as plt

%matplotlib inline

# Построим гистограммы значений

fig = plt.figure(figsize=(15,15)) # создаем объект фигуры

cols = 3                         # определяем количество столбцов (как нам удобно смотреть)

rows = np.ceil(float(df.shape[1]) / cols) # определяем количество строк | np.ceil - округление к большему. Делим количество строк в датафрейме на количество выбранных колонок cols

for i, column in enumerate(df.columns): # для каждого i, column в нумерованном перечне колонок:

    ax = fig.add_subplot(rows, cols, i + 1)       # добавляем оси и номер каждой гистограммы

    ax.set_title(column)                          # добавляем подписи по имени колонок

    df[column].hist()            # по [колонке] строим график. Вызываем метод hist, привязываем его к объекту ax

    plt.xticks(rotation="vertical")               # параметры отображения подписей

plt.subplots_adjust(hspace=0.7, wspace=0.2)       # отображаем
# построим тепловую карту, ищем корреляцию

import seaborn as sns



plt.subplots(figsize=(10,10))

#print(plt.subplots(figsize=(10,10)))

sns.heatmap(df.corr(), square = True, annot = True)     # метод .corr() считает коррекляци, .heatmap - создает тепловую карту

plt.show()                                # отображаем
# Довольно сильно коррелируют данные excessIntProf & skewnessIntProf. Дропнем одну из колонок

df.drop(['skewnessIntProf'], inplace = True, axis = 1)
#преобразуем данные в numpy массив и выделим целевую переменную

X = np.array(df.drop(['class'], axis = 1))

y = np.array(df['class'])
X
y
# нормализуем данные

from sklearn.preprocessing import scale # импортируем модуль scale

X_scaled = scale(np.array(X,dtype = 'float'), with_std = True, with_mean = True) # присваиваем переменной X_scale масштабируемый numpy массив

X_scaled
# Баллансировка данных. Получаем индексы звезд не пульсаров

not_pulsar_index = np.argwhere(y == 0).flatten() # записываем индексы элементов с "0" в массив 

not_pulsar_index, len(not_pulsar_index)
#Находим индексы обрезаемых элементов

from sklearn.utils import shuffle



not_pulsar_index = shuffle(not_pulsar_index, random_state = 42)

not_pulsar_index = not_pulsar_index[len(np.argwhere(y == 1).flatten()):]

not_pulsar_index, len(not_pulsar_index)
#проверим балланс

len(np.argwhere(y == 0).flatten()) - len(not_pulsar_index) == len(np.argwhere(y == 1).flatten())
# обрезаем 

X_scaled = np.delete(X_scaled, not_pulsar_index, 0)

y = np.delete(y, not_pulsar_index, 0)

# отобразим итоговый размер выборок

X_scaled.shape, y.shape
# разделим выборку на тренировочную и тестовую в соотношении 80/20



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.20)
# K ближайших соседей. Юзаем кросс-валидацию для подбора наилучших параметров

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



knn = KNeighborsClassifier()

knn_grid = {'n_neighbors': np.array(np.linspace(15, 25, 15), dtype='int')}

gs = GridSearchCV(knn, knn_grid, cv=5)

# обучаем и смотрим лучшие параметры

gs.fit(X_train, y_train)

gs.best_params_, gs.best_score_

# Функция отрисовки графиков



def grid_plot(x, y, x_label, title, y_label='roc_auc'):

    plt.figure(figsize=(12, 6)) # Создаем фтгуру

    plt.grid(True)              # Создаем сетку

    plt.plot(x, y, 'go-')       # отображаем данные

    plt.xlabel(x_label)         # Подпись по оси X 

    plt.ylabel(y_label)         # Подпись по оси Y

    plt.title(title)            # Заголовок, название графика
# Строим график зависимости качества от числа соседей

# замечание: результаты обучения хранятся в атрибуте cv_results_ объекта gs



grid_plot(knn_grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier')
# запустим еще раз с уточненными параметрами точек

knn = KNeighborsClassifier()

knn_grid = {'n_neighbors': np.array(np.linspace(5, 20, 20), dtype='int')}

gs = GridSearchCV(knn, knn_grid, cv=10)

# обучаем и смотрим лучшие параметры

gs.fit(X_train, y_train)

gs.best_params_, gs.best_score_

# запишем лучшее качество для анализа

best_par = {}

best_par['KNN'] = gs.best_score_
# строим зависимость повторно

grid_plot(knn_grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier')
# обучаем  с лучшими параметрами и предсказываем

model_knn = KNeighborsClassifier(n_neighbors = 10)

model_knn.fit(X_train, y_train)



y_knn = model_knn.predict(X_test)
# посмотрим на метрики качества

from sklearn import metrics



print ('knn.accuracy =', metrics.accuracy_score(y_test, y_knn), 'knn.f1_score =', metrics.f1_score(y_test, y_knn))

cr = metrics.classification_report(y_test, y_knn)

print(cr)
# используем случайный лес

from sklearn.ensemble import RandomForestClassifier



param_grid = {'n_estimators': [i for i in range(2, 50)]} # определеим диапазон количества деревьев



alg = RandomForestClassifier()

gs = GridSearchCV(alg, param_grid, cv=5)

gs.fit(X_train, y_train)

# построим график

grid_plot(param_grid['n_estimators'], gs.cv_results_['mean_test_score'], 'n_estimators', 'forest')
# отобразим лучшие параметры. используем их для обучения

print(gs.best_params_, gs.best_score_)

model_forest = RandomForestClassifier(n_estimators = gs.best_params_['n_estimators'])

model_forest.fit(X_train, y_train)



y_forest = model_forest.predict(X_test)
best_par['RandomForest'] = gs.best_score_
# посмотрим на метрики качества

print(metrics.classification_report(y_test, y_forest))

# воспользуемся алгоритмом логистической регрессии. Все то же самое, только другой классификатор:

from sklearn.linear_model import LogisticRegression



param_grid = {'penalty': ['l1', 'l2'],'C': np.array(np.linspace(1, 20, 20), dtype='int')}

lc = LogisticRegression()

gs = GridSearchCV(lc, param_grid, cv=5)

# обучаем

gs.fit(X_train, y_train)



gs.best_params_, gs.best_score_

best_par['LogisticRegression'] = gs.best_score_

model_lr = LogisticRegression(penalty = gs.best_params_['penalty'], C = gs.best_params_['C'])

model_lr.fit(X_train, y_train)



y_forest = model_lr.predict(X_test)
import pylab



print(best_par)



fig = plt.figure(figsize=(8,6))

plt.bar (range(len(best_par)), best_par.values(), align = 'center')



plt.ylabel('Scores')

plt.xticks(range(len(best_par)),best_par.keys())

plt.legend()

plt.title('Сравнение')

pylab.ylim(0.85, 0.980)



plt.show()