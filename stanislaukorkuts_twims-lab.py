import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from scipy.stats import chi2
series = np.array(

    [53.7, 59.8, 40.8, 31.4, 51.9, 63.3, 57.7, 53.8, 31.0, 62.8,

     48.4, 49.4, 58.1, 38.9, 27.6, 30.5, 46.8, 48.2, 38.3, 41.4,

     35.5, 39.0, 71.6, 45.9, 54.6, 44.0, 52.4, 49.0, 47.4, 48.2,

     48.3, 69.6, 45.7, 44.5, 60.9, 65.7, 53.6, 72.5, 69.2, 51.5,

     52.3, 48.4, 28.9, 64.5, 60.8, 64.1, 38.3, 33.9, 44.7, 49.0,

     56.4, 64.4, 72.3, 41.9, 39.4, 47.5, 48.6, 54.2, 29.0, 43.9,

     34.3, 64.4, 44.9, 47.3, 41.6, 56.4, 38.3, 45.5, 52.7, 50.3,

     52.0, 54.1, 53.7, 65.4, 43.5, 60.0, 49.6, 44.4, 40.8, 57.2,

     53.7, 52.6, 47.6, 40.2, 37.4, 40.3, 51.8, 54.3, 61.9, 53.2,

     53.9, 38.0, 50.9, 63.4, 56.5, 61.3, 42.1, 38.6, 53.2, 46.8])
remmembered_series = series.copy()
series = np.sort(series)

print("Вариационный ряд")

print(series)
x_min = np.min(series)

x_max = np.max(series)

x_mean = np.round(np.mean(series), 3)

h = (x_max - x_min) / 10

D = np.round(np.sum((series - x_mean) ** 2) / 99, 3)

sigm = np.round(np.sqrt(D), 3)
print(f'Минимальный x = {x_min}\n' +

      f'Максимальный x = {x_max}\n' +

      f'Среднее x = {x_mean}\n' +

      f'Ширина интервала группирования = {h}\n' +

      f'Выборочная дисперсия = {D}\n' +

      f'Стандартное отклонение = {sigm}\n')
data = []

w_sum = 0

for i in range(10):

    x0 = x_min + i * h

    x1 = x_min + (i + 1) * h

    x_avg = (x1 + x0) / 2

    m = len([value for value in series if value < x1 and value >= x0])

    if i == 9:

        m += 1

    w = m / 100

    w_sum += w

    data.append([x0, x1, x_avg, m, w, w_sum])
data = pd.DataFrame(data)

data.columns = ['Начало интервала', 'Конец интервала', 'Середина интервала', 'm', 'w', 'Сумма w']

data
plt.plot(data['Середина интервала'], data['w'])

plt.xlabel('Интервал')

plt.ylabel('Относительная частота');
plt.bar(data['Середина интервала'], data['w'], align='center', width=4)

plt.xlabel('Интервалы')

plt.ylabel('Относительная частота');
plt.plot(data['Середина интервала'], data['Сумма w'])
new_data = data[['Начало интервала', 'Конец интервала', 'm']].copy()

new_data['Начало норм. интервала'] = (data['Начало интервала'] - x_mean) / sigm

new_data['Конец норм. интервала'] = (data['Конец интервала'] - x_mean) / sigm

new_data['Начало норм. интервала'][0] = '-inf'

new_data['Конец норм. интервала'][9] = '+inf'
new_data
p1 = 0.5 * (-0.9109 + 1)

p2 = 0.5 * (-0.7959 + 0.9109)

p3 = 0.5 * (-0.5991 + 0.7959)

p4 = 0.5 * (-0.3182 + 0.5991)

p5 = 0.5 * (0.0160 + 0.3182)

p6 = 0.5 * (0.3545 - 0.0160)

p7 = 0.5 * (0.6265 - 0.3545)

p8 = 0.5 * (0.8132 - 0.6265)

p9 = 0.5 * (0.9216 - 0.8132)

p10 = 0.5 * (1 - 0.9216)
new_data['p'] = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])

print(f"Сумма p = {round(np.sum(new_data['p']), 3)}")

new_data
new_data['np'] = new_data['p'] * 100

new_data['(m - np)^2'] = (new_data['m'] - new_data['np']) ** 2

new_data['(m - np)^2 / np'] = new_data['(m - np)^2'] / new_data['np']

xi_square = np.sum(new_data['(m - np)^2 / np'])

print(f'Значения критерия хи-квадрат = {xi_square}')

new_data
print('Нет оснований отвергать гипотезу о нормальном распределении') if xi_square < 14.067 else print(

                                                                'Есть основания отвергать гипотезу о нормальном распределении')
beta = 0.95

alpha = 0.05
t = 1.984

coef = (t * sigm / 10)
print(f'Доверительный интервал для мат. ожидания ({round(x_mean - coef, 3)}, {round(x_mean + coef, 3)})')
alpha1 = (1 - beta) / 2

alpha2 = (1 + beta) / 2

nu = 99

print(round(alpha1, 3), alpha2, nu)
xi_square1 = chi2.ppf(alpha1, nu)

xi_square2 = chi2.ppf(alpha2, nu)

print(xi_square2, xi_square1)
print('Доверительный интервал для среднеквадратического отклонения ' +

      f'({round(np.sqrt(99) * sigm / np.sqrt(xi_square2), 3)}, {round(np.sqrt(99) * sigm / np.sqrt(xi_square1), 3)})')
x_lambda = round(np.sum(remmembered_series[:20]) / 200, 2)

print(f"Оценка параметра лямбда по выборке выборке x от 1 до 20 с n = 200: {x_lambda}")
print(f"Оценка параметра p: {round(np.sum(remmembered_series[:10] + remmembered_series[20:30]) / 200 / 200, 3)}")
print(f"Оценка параметра lambda: {round(1 / x_lambda, 3)}")
print(f"Оценка параметра a = {round(np.sum(remmembered_series[:10] + remmembered_series[20:30]) / 200, 3)}. Оценка параметра sigm = {round(np.std(remmembered_series[:10] + remmembered_series[20:30]), 3)}")
sigm = np.sqrt(109) # Вычисляю стандартное отклонение
t = 1.656

print(f"Доверительный интервал для мат ожидания = ({round(110 - t * sigm / np.sqrt(138), 3)}, {round(110 + t * sigm / np.sqrt(138), 3)})")
D = (76 * 2.8) / 75

sigm = np.sqrt(D)
t = 1.665
print(f"Доверительный интервал для мат ожидания = ({round(3.4 - t * sigm / np.sqrt(76), 3)}, {round(3.4 + t * sigm / np.sqrt(76), 3)})")
sigm = np.sqrt(14.5)

nu = 19

alpha1 = 0.01

alpha2 = 0.99
xi_square1 = chi2.ppf(alpha1, nu)

xi_square2 = chi2.ppf(alpha2, nu)
print('Доверительный интервал для среднеквадратического отклонения ' +

      f'({round(np.sqrt(19) * sigm / np.sqrt(xi_square2), 3)}, {round(np.sqrt(19) * sigm / np.sqrt(xi_square1), 3)})')
x = np.array([16, 22, 24, 35, 31, 54, 59, 75, 80, 81])

y = np.array([3961, 3520, 2960, 2111, 2460, 1693, 1582, 620, 20, 1689])
plt.scatter(x, y)
x_mean = np.mean(x)

y_mean = np.mean(y)

x_square_mean = np.mean(x ** 2)

y_square_mean = np.mean(y ** 2)

xy_mean = np.mean(x * y)

Dx = x_square_mean - x_mean ** 2

Dy = y_square_mean - y_mean ** 2
b = (xy_mean - x_mean * y_mean) / Dx

a = y_mean - b * x_mean
r = (xy_mean - x_mean * y_mean) / (np.sqrt(Dx) * np.sqrt(Dy))
plt.scatter(x, y)

plt.plot([np.min(x), np.max(x)], [a + b * np.min(x), a + b * np.max(x)])
print(f"Коэффициент корреляции равен {r}")