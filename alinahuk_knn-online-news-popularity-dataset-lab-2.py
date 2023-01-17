import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set();
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv') 

df.head()
df.info()
df.describe().T
df['shares']
sns.kdeplot(df['shares'], color="crimson")
from scipy.stats import normaltest



df['shares_log'] = np.log(df['shares']) # создаем новый пригнак - логарифмированный признак shares

stat, p = normaltest(df['shares_log'])

print('p-value:', p)
df1 = df.drop(['shares'], axis=1) # удалим shares
df1 = df.drop(['url'], axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()





# Создание X, y

# X --- вся таблица без таргета

# y --- таргет (целевая переменная)

X = df1.drop('shares_log', axis = 1)

y = df1['shares_log']
X_new = scaler.fit_transform(X)

X_new
from sklearn.model_selection import train_test_split



# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации

# random_state --- произвольное целое число, для воспроизводимости случайных результатов

X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, test_size=0.25, random_state=42)
# хотелость бы отметить, что масштабирование можно было бы выполнить уже на данном этапе, но это не сильно повлияло бы на результат, поэтому я решила оставить его до разбиения

#X_train_scaled = scaler.fit_transform(X_train)

#X_valid_scaled = scaler.transform(X_valid)
from sklearn.neighbors import KNeighborsRegressor



#n_jobs=-1 выбирает все физические ядра и максимально увеличивает их использование

knn = KNeighborsRegressor(n_neighbors=20, n_jobs = -1)
knn.fit(X_train, y_train) #обучение (fit) модели на X_train, y_train
y_pred = knn.predict(X_valid) # предсказание (predict) для X_valid

y_pred
knn.score(X_valid, y_valid)
from sklearn.metrics import mean_squared_error



mean_squared_error(y_valid, y_pred)
from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=True, random_state=42) 
from sklearn.model_selection import cross_val_score



# explained_variance

scores = cross_val_score(knn, X_new, y, cv=kf, scoring='explained_variance')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))
# max_error

scores = cross_val_score(knn, X_new, y, cv=kf, scoring='max_error')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))
# neg_mean_squared_error

scores = cross_val_score(knn, X_new, y, cv=kf, scoring='neg_mean_squared_error')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))
# r2

scores = cross_val_score(knn, X_new, y, cv=kf, scoring='r2')

print('Массив значений метрики:', scores)

print('Средняя метрика на кросс-валидации:', np.mean(scores))
from sklearn.model_selection import GridSearchCV



knn_params={'n_neighbors': np.arange(1, 51)} # словарь параметров (ключ: набор возможных значений)



knn_grid = GridSearchCV(knn, knn_params, cv=kf, scoring='explained_variance', n_jobs = -1)

knn_grid.fit(X_train, y_train)
knn_grid.cv_results_
print("При параметре k =", knn_grid.best_params_.get('n_neighbors'),"качество модели наилучшее.")
knn_grid.best_score_

print(knn_grid.best_score_,"- наилучшая оценка качества модели.")
results_df = pd.DataFrame(knn_grid.cv_results_)

h = 'mean_test_score' # правильность

v = 'param_n_neighbors' #количество соседей

plt.plot(results_df[h], results_df[v], color="crimson")



plt.xlabel(h)

plt.ylabel(v)

plt.title('Validation curve')

plt.show()
p_params = {'p': np.linspace(1, 10, num = 200)}
# weights='distance' –– добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей

knn1 = KNeighborsRegressor(n_neighbors=18, weights='distance', n_jobs = -1)

knn1
knn1_grid = GridSearchCV(knn1, p_params, cv=kf, scoring='explained_variance', n_jobs = -1)

knn1_grid.fit(X_train, y_train)
knn1_grid.cv_results_
#results_df['mean_test_score'].max()
cv_results = pd.DataFrame(knn1_grid.cv_results_)

v = 'mean_test_score'

h = 'param_p'

plt.plot(cv_results[h], cv_results[v], color="crimson")



plt.xlabel(h)

plt.ylabel(v)

plt.title('Validation curve')

plt.show()
print("Наилучшая оценка качества модели: ", knn1_grid.best_score_ )
from sklearn.neighbors import RadiusNeighborsRegressor

rnn = RadiusNeighborsRegressor(radius = 50, n_jobs = -1) #берем большой радиус, что у всех точно были соседи и мы не столкнулись с nan

rnn.fit(X_train, y_train)
y_pred = rnn.predict(X_valid)

y_pred
rnn.score(X_valid, y_valid)
mean_squared_error(y_valid, y_pred)