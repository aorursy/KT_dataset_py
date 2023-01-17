# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

file_path = '/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv'

data = pd.read_csv(file_path)

data.describe()



# Any results you write to the current directory are saved as output.
data.columns
target = data[' Mean of the integrated profile']

sns.distplot(target)

plt.show()
target = data[' Mean of the integrated profile']

sns.distplot(np.log(target))

plt.show()
fig = plt.figure(figsize=(10, 5))



fig.add_subplot(1, 2, 1)

sns.distplot(data[' Excess kurtosis of the integrated profile'])



fig.add_subplot(1, 2, 2)

sns.distplot(data[' Standard deviation of the DM-SNR curve'])



plt.tight_layout()

plt.show()
print(data.isnull().sum())

min1 = data[' Excess kurtosis of the integrated profile'][0]



for r in data[' Excess kurtosis of the integrated profile']:

    if(min1 > r):

        min1 = r

print(min1)



for i in range(len(data[' Excess kurtosis of the integrated profile'])):

    data[' Excess kurtosis of the integrated profile'][i] -= min1 - 1
fig = plt.figure(figsize=(10, 5))



fig.add_subplot(1, 2, 1)

sns.distplot(np.log(data[' Excess kurtosis of the integrated profile']))



fig.add_subplot(1, 2, 2)

sns.distplot(np.log(data[' Standard deviation of the DM-SNR curve']))

    

plt.tight_layout()

plt.show()
main = np.log(data[' Standard deviation of the DM-SNR curve'])



target_log = np.log(target)
fig = plt.figure(figsize=(10, 10))



fig.add_subplot(2, 2, 1)



sns.scatterplot(data[' Standard deviation of the DM-SNR curve'], target)



fig.add_subplot(2, 2, 2)

sns.scatterplot(main, target)



fig.add_subplot(2, 2, 3)

sns.scatterplot(data[' Standard deviation of the DM-SNR curve'], target_log)



fig.add_subplot(2, 2, 4)

sns.scatterplot(main, target_log)



plt.tight_layout()

plt.show()
min_x = main.min()

max_x = main.max()
t0 = 0

t1 = 1

hmn = t0 + t1 * min_x

hmx = t0 + t1 * max_x



fig, ax = plt.subplots()



sns.scatterplot(main, target_log, ax=ax)



sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)
J = 0

for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

    J += (np.exp(t0 + t1 * main[i]) - data[' Mean of the integrated profile'][i])**2

J /= 2 * len(data[' Standard deviation of the DM-SNR curve'])

print(J)
sns.regplot(main, target_log)

plt.show()
# t1_arr = []

# t2_arr = []

# J_arr = []

# best_t1 = -1

# best_t2 = -1

# best_J = 1000000

# for i in np.arange(-2.5, 2.5, 0.1):

#     for j in np.arange(0.0, 2.0, 0.1):

#         tmp_J = 0

#         for k in range(len(data[' Standard deviation of the DM-SNR curve'])):

#             tmp_J += (np.exp(i + j * main[k]) - data[' Mean of the integrated profile'][k])**2

#         tmp_J /= 2 * len(data[' Standard deviation of the DM-SNR curve'])

#         if tmp_J < best_J:

#             best_i = i

#             best_j = j

#             best_J = tmp_J

#         t1_arr.append(i)

#         t2_arr.append(j)

#         J_arr.append(min(tmp_J, 10000))

# print(best_J)

# print(i)

# print(j)
# import plotly.offline as py

# import plotly.graph_objs as go

# py.init_notebook_mode(connected=True)



# trace1 = go.Scatter3d(

#     x=t1_arr,

#     y=t2_arr,

#     z=J_arr,

#     mode='markers',

#     marker=dict(

#         size=5,

#         line=dict(

#             color='rgba(217, 217, 217, 0.14)',

#             width=0.1

#         ),

#         opacity=1

#     ),

#     name = 'price'

# )

# fig = go.Figure(data=trace1)

# py.iplot(fig)
t0_best = t0

t1_best = t1

hmn = t0_best + t1_best * min_x

hmx = t0_best + t1_best * max_x



# Построим два графика на одном

fig, ax = plt.subplots()

# декартова плоскость с размеченными точками

sns.scatterplot(main, target_log, ax=ax)

# прямая, соответствующая функции h

sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)

plt.show()
# Возьмём шаг обучения 0.01

t0_best = t0

t1_best = t1



alpha = 0.01

# желаемая точность

eps = 0.002

# количество шагов, которое понадобилось сделать для достижения желаемой точности

c = 1



# Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

J_t0 = 0

for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

    J_t0+= (t0_best + t1_best * main[i] - target_log[i])

# делим на количество

J_t0 /= len(data[' Standard deviation of the DM-SNR curve'])

# умножаем на шаг обучения

J_t0 *= alpha



# Высляем частную производную функции J по t_1 во всех точках выборки, суммируем

J_t1 = 0

for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

    J_t1 += (t0_best + t1_best * main[i] - target_log[i]) * main[i]

# делим на количество

J_t1 /= len(data[' Standard deviation of the DM-SNR curve'])

# умножаем на шаг обучения

J_t1 *= alpha



# обновляем значения t0 и t1

t0_best = t0_best - J_t0

t1_best = t1_best - J_t1



# запускаем цикл, пока J_t0 и J_t1 больше желаемой точности

# значения J_t0 и J_t1 показывают, на сколько изменились t0_best и t1_best

# и если это изменение меньшн eps, то мы достигли нужной точности

while (abs(J_t0) > eps or abs(J_t1) > eps):

    # Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

    J_t0 = 0

    for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

        J_t0+= (t0_best + t1_best * main[i] - target_log[i])

    # делим на количество

    J_t0 /= len(data[' Standard deviation of the DM-SNR curve'])

    # умножаем на шаг обучения

    J_t0 *= alpha



    # Высляем частную производную функции J по t_1 во всех точках выборки, суммируем

    J_t1 = 0

    for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

        J_t1 += (t0_best + t1_best * main[i] - target_log[i]) * main[i]

    # делим на количество

    J_t1 /= len(data[' Standard deviation of the DM-SNR curve'])

    # умножаем на шаг обучения

    J_t1 *= alpha

    # обновляем значения t0 и t1

    t0_best = t0_best - J_t0

    t1_best = t1_best - J_t1

    c += 1

    print(c)

    print("{0} {1}".format(J_t0, J_t1))

    print("{0} {1}".format(t0_best, t1_best))

# выводим на экран количество шагов

print(c)

# выводим на экрнан значения J_t0 и J_t1, чтобы убедиться в достигнутой точности

print(J_t0, J_t1)

# выводим на экран лучшие значения t0 и t1

print(t0_best, t1_best)

# отображаем график

hmn = t0_best + t1_best * min_x

hmx = t0_best + t1_best * max_x

fig, ax = plt.subplots()

sns.scatterplot(main, target_log, ax=ax)

sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)

plt.show()
# Вычислим значение J в точке начального приближения t0, t1 

# в моём случае (0, 1)

# если работа шла с логарифмическими величинами, то нужно не забыть взять экспоненту e^(h(x))

# если вы работали с исходными величинами, то np.exp в вашем случае не будет

J = 0

for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

    J += (np.exp(t0 + t1 * main[i]) - data[' Mean of the integrated profile'][i])**2

J /= 2 * len(data[' Standard deviation of the DM-SNR curve'])

print(J)



# Вычислим значение J в точке t0_best, t1_best

# аналогично если работа шла с логарифмическими величинами, то нужно не забыть взять экспоненту e^(h(x))

# если вы работали с исходными величинами, то np.exp в вашем случае не будет

J = 0

for i in range(len(data[' Standard deviation of the DM-SNR curve'])):

    J += (np.exp(t0_best + t1_best * main[i]) - data[' Mean of the integrated profile'][i])**2

J /= 2 * len(data[' Standard deviation of the DM-SNR curve'])

print(J)
# Построим графики других параметром, чтобы выбрать пригодные для построения регрессии

attribs = data.select_dtypes(exclude='object').drop(' Mean of the integrated profile', axis=1).copy()



# построим все 10 графиков на одной фигуре

fig = plt.figure(figsize=(20,25))

for i in range(len(attribs.columns)):

    fig.add_subplot(4, 3, i+1)

    # перед построением удаляем NaN значения

    sns.distplot(attribs.iloc[:, i].dropna())

    plt.xlabel(attribs.columns[i])

    

plt.tight_layout()

plt.show()
features = [' Skewness of the integrated profile', ' Excess kurtosis of the integrated profile', ' Mean of the DM-SNR curve', ' Standard deviation of the DM-SNR curve', ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']

print(len(features)) # 8

theta0 = 0

thetaJ = [1 for i in range(len(features))]

# thetaJ = [1, 1, 1, 1, 1, 1, 1, 1]
# Вычислим значение J

thetaJnp = np.array(thetaJ)

X = data[features]

#print(X)

htheta = thetaJnp.dot(np.transpose(X))

J = 0

for i in range(len(htheta)):

    J += (t0 + htheta[i] - data[' Mean of the integrated profile'][i])**2

J /= 2 * len(htheta)

print(J)
# Возьмём шаг обучения 0.0001

alpha = 0.00001

# желаемая точность

eps = 0.001

# количество шагов, которое понадобилось сделать для достижения желаемой точности

c = 1



X = data[features]



# задаём начальное приближение

theta0_best = 0

theta_best = [1 for i in range(len(features))]

htheta_best = np.array(theta_best).dot(np.transpose(X))



cont = True

# запускаем цикл, пока хотя бы одно значение частной производной больше eps

while True:  

    # Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

    J_theta0 = 0

    for i in range(len(htheta_best)):

        J_theta0 += (theta0_best + htheta_best[i] - data[' Mean of the integrated profile'][i])

    # делим на количество

    J_theta0 /= len(htheta_best)

    # умножаем на шаг обучения

    J_theta0 *= alpha



    J_theta = [0 for i in range(len(features))]

    for j in range(len(features)):

        # Высляем частную производную функции J по t_j во всех точках выборки, суммируем

        J_theta[j] = 0

        for i in range(len(htheta_best)):

            J_theta[j] += (theta0_best + htheta_best[i] - data[' Mean of the integrated profile'][i]) * data[features[j]][i]

        # делим на количество

        J_theta[j] /= len(htheta_best)

        # умножаем на шаг обучения

        J_theta[j] *= alpha

    # обновляем значение коэффициентов тета

    theta0_best = theta0_best - J_theta0

    theta_best = np.subtract(theta_best, J_theta)

    htheta_best = np.array(theta_best).dot(np.transpose(X))

    # пересчитываем оценочную функцию

    J = 0

    for i in range(len(htheta_best)):

        J += (theta0_best + htheta_best[i] - data[' Mean of the integrated profile'][i])**2

    J /= 2 * len(htheta_best)

    # Выводим на экран J, чтобы убедиться, что оценочная функция улучшается

    # Если при выводе значения будут увеличиваться, то цикл нужно остановить и скоректировать alpha

    print(J)

    # Для примера делаем отсечение в 200 шагов, иначе слишком долго сходится

    c += 1

    if c == 100:

        break

    # значения J_theta0 и J_theta показывают, на сколько изменились theta0_best и theta_best

    # и если это изменение меньшн eps, то мы достигли нужной точности

    continue_loop = False

    for i in range(len(J_theta)):

        if abs(J_theta[i]) > eps:

            continue_loop = True

            break

    continue_loop = continue_loop or abs(J_theta0) > eps

    if not continue_loop:

        break   



print('t0 =', theta0_best)

for i in range(len(theta_best)):

    print('t', i + 1, ' = ', theta_best[i], sep="")
attribs = data.select_dtypes(exclude='object').drop(' Mean of the integrated profile', axis=1).copy()



# построим все 10 графиков на одной фигуре

fig = plt.figure(figsize=(20,25))

for i in range(len(attribs.columns)):

    fig.add_subplot(4, 3, i+1)

    # перед построением удаляем NaN значения

    sns.distplot(attribs.iloc[:, i].dropna())

    plt.xlabel(attribs.columns[i])

    

plt.tight_layout()

plt.show()
features = [' Skewness of the integrated profile', ' Excess kurtosis of the integrated profile', ' Mean of the DM-SNR curve', ' Standard deviation of the DM-SNR curve', ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']

X_corr = data[features + [' Mean of the integrated profile']]

X_corr.describe()



x_corr_info =  X_corr.corr()

f, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(x_corr_info, annot=True)

plt.show()
features = [' Skewness of the integrated profile', ' Excess kurtosis of the integrated profile', ' Mean of the DM-SNR curve', ' Standard deviation of the DM-SNR curve']



print(len(features))

theta0 = 0

thetaJ = [1 for i in range(len(features))]
thetaJnp = np.array(thetaJ)

X = data[features]

htheta = thetaJnp.dot(np.transpose(X))

J = 0

for i in range(len(htheta)):

    J += (t0 + htheta[i] - data[' Mean of the integrated profile'][i])**2

J /= 2 * len(htheta)

print(J)
alpha = 0.00022

# желаемая точность

eps = 0.001

# количество шагов, которое понадобилось сделать для достижения желаемой точности

c = 2



X = data[features]



# задаём начальное приближение

theta0_best = 0

theta_best = [1 for i in range(len(features))]

htheta_best = np.array(theta_best).dot(np.transpose(X))



cont = True

# запускаем цикл, пока хотя бы одно значение частной производной больше eps

while True:  

    # Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

    J_theta0 = 0

    for i in range(len(htheta_best)):

        J_theta0 += (theta0_best + htheta_best[i] - data[' Mean of the integrated profile'][i])

    # делим на количество

    J_theta0 /= len(htheta_best)

    # умножаем на шаг обучения

    J_theta0 *= alpha



    J_theta = [0 for i in range(len(features))]

    for j in range(len(features)):

        # Высляем частную производную функции J по t_j во всех точках выборки, суммируем

        J_theta[j] = 0

        for i in range(len(htheta_best)):

            J_theta[j] += (theta0_best + htheta_best[i] - data[' Mean of the integrated profile'][i]) * data[features[j]][i]

        # делим на количество

        J_theta[j] /= len(htheta_best)

        # умножаем на шаг обучения

        J_theta[j] *= alpha



    # обновляем значение коэффициентов тета

    theta0_best = theta0_best - J_theta0

    theta_best = np.subtract(theta_best, J_theta)

    htheta_best = np.array(theta_best).dot(np.transpose(X))

    # пересчитываем оценочную функцию

    J = 0

    for i in range(len(htheta_best)):

        J += (theta0_best + htheta_best[i] - data[' Mean of the integrated profile'][i])**2

    J /= 2 * len(htheta_best)

    # Выводим на экран J, чтобы убедиться, что оценочная функция улучшается

    # Если при выводе значения будут увеличиваться, то цикл нужно остановить и скоректировать alpha

    print(J)

    # Для примера делаем отсечение в 200 шагов, иначе слишком долго сходится

    c += 1

    if c == 100:

        break

    # значения J_theta0 и J_theta показывают, на сколько изменились theta0_best и theta_best

    # и если это изменение меньшн eps, то мы достигли нужной точности

    continue_loop = False

    for i in range(len(J_theta)):

        if abs(J_theta[i]) > eps:

            continue_loop = True

            break

    continue_loop = continue_loop or abs(J_theta0) > eps

    if not continue_loop:

        break   



print('t0 =', theta0_best)

for i in range(len(theta_best)):

    print('t', i + 1, ' = ', theta_best[i], sep="")
X1 = data[features][:len(data[features]) // 2]

X2 = data[features][len(data[features]) // 2:]

y1 = data[' Mean of the integrated profile'][:len(data[features]) // 2]

y2 = data[' Mean of the integrated profile'][len(data[features]) // 2:]
from sklearn.linear_model import LinearRegression

# Обучаем модель на наборе X1 y1

lreg = LinearRegression().fit(X1, y1)

print(lreg.score(X1, y1))

print(lreg.score(X2, y2))
# Находим значение функции J на наборе X1

y_pred1 = lreg.predict(X1)

J = 0

for i in range(len(y_pred1)):

    J += (y1[i] - y_pred1[i])**2

J /= 2 * len(y_pred1)

print(J)

# Находим значение функции J на наборе X2

y_pred2 = lreg.predict(X2)

J = 0

for i in range(len(y_pred2)):

    J += (y2[i + len(y_pred1)] - y_pred2[i])**2

J /= 2 * len(y_pred2)

print(J)
from sklearn.ensemble import GradientBoostingRegressor

# Обучаем модель на наборе X1 y1

gbr = GradientBoostingRegressor(learning_rate=0.02).fit(X1, y1)

print(gbr.score(X1, y1))

print(gbr.score(X2, y2))
# Находим значение функции J на наборе X1

y_pred_gbr1 = gbr.predict(X1)

J = 0

for i in range(len(y_pred_gbr1)):

    J += (y1[i] - y_pred_gbr1[i])**2

J /= 2 * len(y_pred_gbr1)

print(J)

# Находим значение функции J на наборе X2

y_pred_gbr2 = gbr.predict(X2)

J = 0

for i in range(len(y_pred_gbr2)):

    J += (y2[i + len(y_pred_gbr1)] - y_pred_gbr2[i])**2

J /= 2 * len(y_pred_gbr2)

print(J)
from sklearn.ensemble import AdaBoostRegressor

# Обучаем модель на наборе X1 y1

ada = AdaBoostRegressor().fit(X1, y1)

print(ada.score(X1, y1))

print(ada.score(X2, y2))
# Находим значение функции J на наборе X1

y_pred_ada1 = ada.predict(X1)

J = 0

for i in range(len(y_pred_ada1)):

    J += (y1[i] - y_pred_ada1[i])**2

J /= 2 * len(y_pred_ada1)

print(J)

# Находим значение функции J на наборе X2

y_pred_ada2 = ada.predict(X2)

J = 0

for i in range(len(y_pred_ada2)):

    J += (y2[i + len(y_pred_ada1)] - y_pred_ada2[i])**2

J /= 2 * len(y_pred_ada2)

print(J)