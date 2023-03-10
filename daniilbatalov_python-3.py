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
print("Задание №6")
number = input("Введите номер дня недели: ") # Запрашивается номер дня недели (в строковом типе)
if number.isdecimal() == True: # Проверяется, будет ли введеная строка являться типом int после её конвертирования в данный тип
    if int(number)<1 or int(number)>7: # Проверяется условие нахождения number перед числом 1 или после числа 7
        print("Такого дня недели не существует!") # Вывод на экран сообщения, если условия верно
    else: # Если условие неверно, то выполняется код этой ветки
        if number == "1": # Если в переменной number записано 1, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Понедельник") # Вывод на экран
        elif number == "2": # Если в переменной number записано 2, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Вторник") # Вывод на экран
        elif number == "3": # Если в переменной number записано 3, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Среда") # Вывод на экран
        elif number == "4": # Если в переменной number записано 4, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Четверг") # Вывод на экран
        elif number == "5": # Если в переменной number записано 5, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Пятница") # Вывод на экран
        elif number == "6": # Если в переменной number записано 6, то выполняется дальнейший код в условии, иначе осуществляется переход к конструкции else
            print("Суббота") # Вывод на экран
        else: # Выполняется, если ни одно условие до этого не было выполнено
            print("Воскресенье") # Вывод на экран
else: # Ветка срабатывает, если введеная строка number не будет являться типом int после её конвертирования в данный тип
    print("Введены некорректные данные!") # Вывод на экран
print("Задание №7")
number = input("Введите номер месяца в году: ") # Запрашивается номер месяца в году (в строковом типе)
if number.isdecimal() == True:# Проверяется, будет ли введеная строка являться типом int после её конвертирования в данный тип
    if int(number)<1 or int(number)>12:# Проверяется условие нахождения number перед числом 1 или после числа 12
        print("Такого месяца не существует!") # Вывод на экран сообщения, если условия верно
    else: # Если условие неверно, то выполняется код этой ветки
        if int(number)>=3 and int(number)<=5: # Если в переменной number находится число от 3 до 5 включительно, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Весна") # Вывод на экран
        elif int(number)>=6 and int(number)<=8: # Если в переменной number находится число от 6 до 8 включительно, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
            print("Лето") # Вывод на экран
        elif int(number)>=9 and int(number)<=11: # Если в переменной number находится число от 9 до 11 включительно, то выполняется дальнейший код в условии, иначе осуществляется переход к конструкции else
            print("Осень") # Вывод на экран
        else: # Выполняется, если ни одно условие до этого не было выполнено
            print("Зима") # Вывод на экран
else:# Ветка срабатывает, если введеная строка number не будет являться типом int после её конвертирования в данный тип
    print("Введены некорректные данные!") # Вывод на экран
print("Задание №8")
s = float(input("Введите расстояние в метрах, которое необходимо преодолеть: "))  # Запрашивается расстояние в метрах, которое необходимо преодолеть  
t = float(input("Введите время в секундах, оставшееся до встречи: ")) # Запрашивается время в секундах, за которое планируется преодолеть введенное раньше расстояние
v = float(input("Введите планируемую скорость в м/c: ")) # Запрашивается скорость в м/c, с которой планируется преодолеть введенное ранее расстояние
Speed = s/t  # Вычисление необходимой скорости. Она будет являться граничной скоростью
if Speed>v:  # Условие - если введенная скорость меньше посчитанной скорости, то выполняется дальнейший код в условии, иначе осуществляется переход к следующему условию
    print("Вы не успеваете на встречу!") # Вывод на экран
elif Speed == v: # Если введенная скорость совпадает с посчитанной, то выполняется дальнейший код в условии, иначе осуществляется переход к конструкции else
    print("Вы успеваете на встречу в точно назначенное время!") # Вывод на экран
else: # Выполняется, если ни одно условие до этого не было выполнено
    print("Вы точно успеваете на встречу!") # Вывод на экран