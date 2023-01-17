# Цель прогнозируемой модели: по данным из датасета спрогнозировать популярность публикации. 

# Целевая пременная: shares - количественная оценка, отражающая, сколько раз поделились ползователи публикацией.

# Датасет содержит информацию о публикациях на сайтах (новости).

# Данная задача - это задача регресии. Прогноз на основе выборки объектов с различными признаками. На выходе должно получиться вещественное число.



import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv', sep=',')

df.head()
df.head().T
df.info()
# drop'аем (удаляем) ненужную колонку "url". 

df1 = df.drop(['url'], axis=1)

df1.head()
df.describe().T
#Извлекаем целевую пеменную. Ранее мы выясниили, что перед нами задача регрессии.

target = df1['shares']

target
# Логарифмируем целевую перменную.

import seaborn as sns

snsplot = sns.kdeplot(df['shares'], shade=True)

figure = snsplot.get_figure()
# Масштабирование данных.

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

StandardScaler(df1)
# Разделим выборку на обучающую и валидационную (тестовую).

from sklearn.model_selection import train_test_split

X = df1.drop(['min_negative_polarity', 'shares', 'max_negative_polarity'], axis=1)

y = df1['shares']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, random_state=42) 



X_train.head()
X_valid.head()
y_train.head()
y_valid.head()
train_test_split(y, shuffle=False)
# Обучаем алгоритм классификации KNeighborsRegressor.

# Предсказания для валидационного множества.

from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors = 10)

neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_valid)

y_pred

# Обучаем алгоритм регрессии.

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=42, max_depth = 5)

tree.fit(X_train, y_train)
# Строим графические полученное дерево.

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='tree.dot', feature_names=X.columns)

print(open('tree.dot').read()) 

#  Оценка модели валидационной выборки с помошью mean_squared_error.

from sklearn.metrics import mean_squared_error

print("Оценка модели валидационной выборки ",mean_squared_error(y_valid, y_pred))
tree.score(X_valid, y_valid)
# Кросс-валидация и подбор гиперпараметров

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

KFold = KFold(n_splits=5, shuffle=True, random_state=42) 

# Создаем генератор разбиений, который перемешивает выборку перед создани-

#ем блоков ( shuffle=True ). Число блоков n_splits равно 5. Задаем также

#параметр random_state для воспроизводимости результатов. В нашем случае = 12.



# Указываем диапазон гиперпараметров для настройки.

hyper_params = [{'n_features_to_select': list(range(1, 14))}]



# Выполняем поиск по сетке.

# Указываем модель.

lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm)             



# Вызов GridSearchCV

model_cv = GridSearchCV(estimator = rfe, 

                        param_grid = hyper_params, 

                        scoring= 'r2', 

                        cv = KFold, 

                        verbose = 1,

                        return_train_score=True)      



# Соответсвующая модели

model_cv.fit(X_train, y_train)



# cv results

cv_results = pd.DataFrame(model_cv.cv_results_).T

cv_results
# Кросс-валидация и подбор гиперпараметрa

from sklearn.model_selection import GridSearchCV

tree_params = {'min_samples_split': np.arange(2, 11)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='max_error') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
# Кросс-валидация и подбор гиперпараметров max_depth - максимальная глубина дерева,

#mian_samples_leaf - минимальное число объектов в листе, max_features - максимальное число признаков, расстмариваемых при поиске лучшего разбиениия. 

from sklearn.model_selection import GridSearchCV

tree_params1 = {'max_depth': np.arange(3, 31)}

tree_grid1 = GridSearchCV(tree, tree_params1, cv=5, scoring='explained_variance') # кросс-валидация по 5 блокам

tree_grid1.fit(X_train, y_train)



tree_grid1.best_estimator_
# Находим оценку качества модели и число, отражающее максимальную глубину.

max_depth = list(tree_grid1.best_params_.values())[0]

print("Максимальная глубина: ", max_depth)

print("Наилучшая оценка качества модели: ", tree_grid1.best_score_)
tree_params2 = {'min_samples_split': np.arange(3, 31)}

tree_grid2 = GridSearchCV(tree, tree_params2, cv=KFold, scoring='explained_variance') # кросс-валидация 

tree_grid2.fit(X_train, y_train)
tree_grid2.best_estimator_
# Находим оценку качества модели и min_samples_split - минимальное число объектов для разбиения во внутренней вершине.

min_samples_split = list(tree_grid2.best_params_.values())[0]

print("Минимальное число разбиения во внутренней вершине: ", min_samples_split)

print("Наилучшая оценка качества модели: ", tree_grid2.best_score_)
tree_params3 = {'min_samples_leaf': np.arange(3, 31)}

tree_grid3 = GridSearchCV(tree, tree_params3, cv=KFold, scoring='explained_variance') # кросс-валидация 

tree_grid3.fit(X_train, y_train)
tree_grid3.best_estimator_
# Находим оценку качества модели и min_samples_split - минимальное число объектов в листе.

min_samples_leaf = list(tree_grid3.best_params_.values())[0]

print("Минимальное число объектов в листе: ", min_samples_leaf)

print("Наилучшая оценка качества модели: ", tree_grid3.best_score_)
tree_params4 = {'max_features': np.arange(3, 31)}

tree_grid4 = GridSearchCV(tree, tree_params4, cv=KFold, scoring='explained_variance') # кросс-валидация 

tree_grid4.fit(X_train, y_train)
tree_grid4.best_estimator_
# Находим оценку качества модели и ma_features - максимальное число признаков, рассматриваемых при наилучшем разбиении.

max_features = list(tree_grid4.best_params_.values())[0]

print("Минимальное число объектов в листе: ", max_features)

print("Наилучшая оценка качества модели: ", tree_grid4.best_score_)
# Рисуем валидационную кривую

# По оси х --- значения гиперпараметров (param_n_features_to_select)

# По оси y --- значения метрики (mean_test_score)

import matplotlib.pyplot as plt

results_df = pd.DataFrame(model_cv.cv_results_)

plt.plot(results_df['param_n_features_to_select'], results_df['mean_test_score'])

# Подписываем оси и график

plt.xlabel('param_n_features_to_select')

plt.ylabel('Test accuracy')

plt.title('Validation curve')

plt.show()
# Отрисовка графиков

# Из полученных данных можем сделать вывод, что хорошим количество гиперпарамтеров будет:

#max_depth = 5, max_features = 4, min_samples_leaf = 30,min_samples_split = 18.

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, figsize = (20,10))



ax[0, 0].plot(tree_params1['max_depth'], tree_grid1.cv_results_['mean_test_score']) 

ax[0, 0].set_xlabel('max_depth')

ax[0, 0].set_ylabel('Mean accuracy on test set')



ax[0, 1].plot(tree_params2['min_samples_split'], tree_grid2.cv_results_['mean_test_score']) 

ax[0, 1].set_xlabel('min_samples_split')

ax[0, 1].set_ylabel('Mean accuracy on test set')



ax[1, 0].plot(tree_params3['min_samples_leaf'], tree_grid3.cv_results_['mean_test_score']) 

ax[1, 0].set_xlabel('min_samples_leaf')

ax[1, 0].set_ylabel('Mean accuracy on test set')



ax[1, 1].plot(tree_params4['max_features'],tree_grid4.cv_results_['mean_test_score']) 

ax[1, 1].set_xlabel('max_features')

ax[1, 1].set_ylabel('Mean accuracy on test set')

# Построение модели случайного леса с количеством деревьев = 100, максимальной глубтной =3.

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(n_estimators=100, max_depth = 3,random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)

mean_squared_error(y_valid, y_pred)



features = {'f' + str(i + 1):name for (i, name) in zip(range(len(df1.columns)), df1.columns)}

importances = tree.feature_importances_

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

# По полученным данным можем сделать вывод, что модель, построенная с помощью деревьев, лучше модели knn.

# Самым влиятельным ялвялется признак - avg_negative_polarity, ему антагонистом является -  LDA_01, kv_avg_min
from sklearn.model_selection import GridSearchCV

tree_paramss = {'n_estimators': [100]}

tree_gridss = GridSearchCV(rf, tree_paramss, cv=KFold, scoring='explained_variance')

tree_gridss.fit(X_train, y_train)
tree_gridss.best_estimator_
# Находим оценку качества модели и n_estimators - количество деревьев.

n_estimatorstr = list(tree_gridss.best_params_.values())[0]

print("Минимальное число объектов в листе: ", n_estimatorstr)

print("Наилучшая оценка качества модели: ", tree_gridss.best_score_)
# Кросс-валидация и подбор гиперпараметров max_depth - максимальная глубина дерева,

#mian_samples_leaf - минимальное число объектов в листе, max_features - максимальное число признаков, расстмариваемых при поиске лучшего разбиениия, n_estimators - количество деревьев. 

from sklearn.model_selection import GridSearchCV



tree_paramss1 = {'max_depth': np.arange(3, 31)}           

tree_gridss1 = GridSearchCV(tree, tree_params1, cv=5, scoring='explained_variance') # кросс-валидация по 5 блокам

tree_gridss1.fit(X_train, y_train)

tree_gridss1.best_estimator_
# Находим оценку качества модели и число, отражающее максимальную глубину.

max_depthtr = list(tree_gridss1.best_params_.values())[0]

print("Максимальная глубина: ", max_depthtr)

print("Наилучшая оценка качества модели: ", tree_gridss1.best_score_)
tree_paramss2 = {'min_samples_split': np.arange(3, 31)}

tree_gridss2 = GridSearchCV(tree, tree_paramss2, cv=KFold, scoring='explained_variance') # кросс-валидация 

tree_gridss2.fit(X_train, y_train)
tree_gridss2.best_estimator_
# Находим оценку качества модели и min_samples_split - минимальное число объектов для разбиения во внутренней вершине.

min_samples_splittr = list(tree_gridss2.best_params_.values())[0]

print("Минимальное число разбиения во внутренней вершине: ", min_samples_splittr)

print("Наилучшая оценка качества модели: ", tree_gridss2.best_score_)
tree_paramss3 = {'min_samples_leaf': np.arange(3, 31)}

tree_gridss3 = GridSearchCV(tree, tree_paramss3, cv=KFold, scoring='explained_variance') # кросс-валидация 

tree_gridss3.fit(X_train, y_train)
tree_gridss3.best_estimator_
# Находим оценку качества модели и min_samples_split - минимальное число объектов для разбиения во внутренней вершине.

min_samples_leaftr = list(tree_gridss3.best_params_.values())[0]

print("Минимальное число разбиения во внутренней вершине: ", min_samples_leaftr)

print("Наилучшая оценка качества модели: ", tree_gridss3.best_score_)
tree_paramss4 = {'max_features': np.arange(3, 31)}

tree_gridss4 = GridSearchCV(tree, tree_paramss4, cv=KFold, scoring='explained_variance') # кросс-валидация 

tree_gridss4.fit(X_train, y_train)
tree_gridss4.best_estimator_
# Находим оценку качества модели и min_samples_split - минимальное число объектов для разбиения во внутренней вершине.

max_featurestr = list(tree_gridss4.best_params_.values())[0]

print("Минимальное число разбиения во внутренней вершине: ", max_featurestr)

print("Наилучшая оценка качества модели: ", tree_gridss4.best_score_)
# Отрисовка графиков

# Из полученных данных можем сделать вывод, что хорошим количество гиперпарамтеров будет:

#max_depth = 5, max_features = 4, min_samples_leaf = 30,min_samples_split = 18, n_estimators = 

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=3, ncols=2, sharey=False, figsize = (40,40))

ax[0, 0].plot(tree_paramss['n_estimators'], tree_gridss.cv_results_['mean_test_score']) 

ax[0, 0].set_xlabel('max_depth')

ax[0, 0].set_ylabel('Mean accuracy on test set')



ax[0, 1].plot(tree_paramss1['max_depth'], tree_gridss1.cv_results_['mean_test_score']) 

ax[0, 1].set_xlabel('max_depth')

ax[0, 1].set_ylabel('Mean accuracy on test set')



ax[1, 0].plot(tree_paramss2['min_samples_split'], tree_gridss2.cv_results_['mean_test_score']) 

ax[1, 0].set_xlabel('min_samples_split')

ax[1, 0].set_ylabel('Mean accuracy on test set')



ax[1, 1].plot(tree_paramss3['min_samples_leaf'], tree_gridss3.cv_results_['mean_test_score']) 

ax[1, 1].set_xlabel('min_samples_leaf')

ax[1, 1].set_ylabel('Mean accuracy on test set')



ax[2, 0].plot(tree_paramss4['max_features'],tree_gridss4.cv_results_['mean_test_score']) 

ax[2, 0].set_xlabel('max_features')

ax[2, 0].set_ylabel('Mean accuracy on test set')

# Оцениваем важность признаков в данной модели. 

#Визуализируем топ-10 самых полезных признаков с помощью столбчатой диаграммы.

# По данныи результатам делаем вывод, что более влиятельным является признак global_sentiment_polarity, а менее - kw_avg_avg.

import matplotlib.pyplot as plt

features = {'f'+str(i+1):name for (i, name) in zip(range(len(df1.columns)), df1.columns)}

# Важность признаков

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

forest.fit(X_train, y_train)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Строим график важности особенностей леса

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]

# Распечатать рейтинг функций

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
#По данной работе мы можем сделать следующие выводы:

#1. Decision Tree имеет особую проблему - молниеносная переобучаемость.

#2. Random Forest по времени рбработки занимает солидную часть моей жизни:) 

#3. На выходе kNN - модель не самого лучшего качетсва.