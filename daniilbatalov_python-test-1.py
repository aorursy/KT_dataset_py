# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Решение методом прямоугольников")
# Заключается в разбиении площади под фигурой на прямоугольники с равным основанием.
# Далее происходит вычисление площади под фигурой путем сложения всех площадей прямоульников.
# Σf(xi-1)Δx
import math # импортируем библиотеку math
a = float(input("Введите начало интервала: ")) # Нижний порог
b = float(input("Введите конец интервала: ")) # Верхний порог
n = 1000 # Количество прямоугольников (точность)
result_1 = 0 # Переменная, содержащая результат вычислений
h = (b - a) / n # Шаг сетки прямоугольников
for i in range(n): # Цикл от 0 до n. Смысл - перебор всех прямоугольников
    result_1 += math.sin(2*(a + h * i)+5) # Вычисление значения функции в каждой точке шага и их сложение
result_1 *= h # Домножение полученной суммы площадей на шаг сетки
print(round(result_1,2)) # Вывод с округлением

print("Решение методом Монте-Карло")
# Суть метода - ограничить площадь под функцией фигурой, площадь которой мы точно знаем.
# Мы выбираем любую точку на этой фигуре и точно можем определить, попала ли она под линию нашей функции.
# Таким образом, мы получаем количество выбранных точек, количество точек, попавших под линию нашей функции.
# Зная площадь выбранной нами фигуры, через пропорцию находим приближенную площадь фигуры под нашей функцией/
import random # Импортируем библиотеку random
N = 0 # Переменная, содержащая в себе количество попаданий под линию функции
S_abcd = (b-a)*1 # Площадь выбранной нами фигуры
m = 100000 # Переменная, задающая точность вычислений
for _ in range (m): # Цикл для случайного выбора точки, итерации зависят от требуемой точности
    x = random.uniform(a,b) # Случайный выбор координаты Х
    y = random.random() # Случайный выбор координаты Y
    if y <= math.sin(2*x+5): #
        N+=1 # Увеличиваем количество попаданий под линию функции на единицу
result_2 = (N*S_abcd)/m # Высичление площади под функцией
print(round(result_2,2)) # Вывод с округлением