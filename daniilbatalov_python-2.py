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
print("Задание №1")
# программа вводит время и скорость, вычисляет
# и выводит длину пути при равномерном движении
t = float(input("Введите время: ")) # Запрашивается время
v = float(input("Введите скорость: ")) # Запрашивается скорость
s = v*t # Вычисляется расстояние
print("Длина пути = " ,s) # Вывод вычисленного расстояния на экран
print("Задание №2")
lenght = float(input("Введите длину помещения: ")) # Запрашивается длина помещения
weight = float(input("Введите ширину помещения: ")) # Запрашивается ширина помещения
height = float(input("Введите высоту помещения: ")) # Запрашивается высота помещения
print("Площадь пола:", lenght*weight, "м2") # Вычисляется площадь пола помещения и вывод результата на экран
print("Площадь стен: ", (2*height*(lenght+weight)), "м2") # Вычисляется площадь стен помещения и вывод результата на экран
print("Объём помещения: ", lenght*weight*height, "м3") # Вычисляется объем помещения и ввод результата на экран
print("Задание №3")
r1 = float(input("Введите сопротивление первого резистора: ")) # Запрашивается сопротивление первого резистора
r2 = float(input("Введите сопротивление второго резистора: ")) # Запрашивается сопротивление второго резистора
r3 = float(input("Введите сопротивление третьего резистора: ")) # Запрашивается сопротивление третьего резистора
par = round(1/(1/r1+1/r2+1/r3),2) # Вычисляется параллельное сопротивление резисторов. Вычисленное значение округляется до 2 знаков после запятой
pos = r1+r2+r3 # Вычислется последовательное сопротивление резисторов
print("Общее сопротивление цепи при последовательном включении резисторов:", pos, "Ом") # Вывод результата на экран
print("Общее сопротивление цепи при параллельном включении резисторов:", par, "Ом") # Вывод результата на экран
print("Задание №4")
import math # Импортирется библиотека для работы со специальными арифмеическими операциями
n = int(input("Сколько людей нужно перевести?: ")) # Запрашивается количество людей
a = int(input("Какой вместимости будут автобусы?: ")) # Запрашивается вместимость одного автобуса
print("Необходимое количество автобусов:", math.ceil(n/a)) # Подсчитывается и выводится на экран результат
print("Задание №5")
time = int(input("Введите интервал в минутах: ")) # Запрашивается время в минутах
h = time//60 # Вычисляется, сколько полных часов в запрошенном значении минут
m = time%60 # Вычисляется остаток минут от полных часов
print(h,"часов",m,"минут") # Вывод результата на экран
print("Задание №6")
import random # Импортируется библиотека random
time = random.randint(100,1000) # Записывается в переменную time случайное значение от 100 до 1000
h = time//60 # Вычисляется, сколько полных часов в переменной time
m = time%60 # Вычисляется остаток минут от полных часов
print(h,"часов",m,"минут") # Вывод результата на экран