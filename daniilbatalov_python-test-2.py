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
def f(a = -5.5, b = -4.5): # Создаем функцию
    """
    Функция предназначена для вычисления значений функции Y=Sin(2x+5) с шагом (b-a)/10
    и возвращает получившиеся значенияс ключом Х и значением Y в табличном виде.
    Функция принимает два параметра:
        а - начало интервала (по умолчанию -5.5)
        b - конец интервала (по умолчанию -4.5)
    
    """
    import math # Импортируем библиотеку math
    X = (b - a) / 10 # Шаг значений функции
    d = {"X":"Y"} # Создаем словарь с условным обозначением строчек
    for i in range(11): # Цикл для заполнения значений и ключей функции Sin(2x+5) в словарь
        result = math.sin(2*(a+X*i)+5) # Вычисление value
        d[round((a+X*i),1)] = round(result,4) # Добавление key/value в словарь
    # Далее описан код, создающий динамически изменяющуюся таблицу
    # по полученным из словаря данным.
    # На мой взгляд, целесообразно перенести этот код в отдельную
    # функцию, которая строит эту таблицу. Параметром функции необходимо
    # сделать словарь.
    dlina_value = 0 # Переменная, в которой хранится длина самого длинного value
    for _, items in d.items(): # Цикл, перебирающий value всего словаря
        if (len(str(items))) > dlina_value: # Ищем самое длинное value
            dlina_value = len(str(items)) # Записываем длину value в переменную
    print('+-----+'+('-'*(dlina_value+2)+'+')*10+'-'*(dlina_value+2)+'+') # Печатаем первую строку таблицы
    for key, _ in d.items(): # Цикл для создания и заполнения второй строки таблицы
        if key == "X": # Условие - если ключ = Х, то выполняется дальнейший код программы
            print("| ",key," ",end="") # Создание ячейки со значением в ней X
        else: # Эта ветка выполняется, если условие ключ = Х, неверно
            if len(str(key)) == 5: # Сравнивает длину ключа с фиксированным значением для корректроного отображения ключа в ячейке. Фиксированное число зависит от округления ключа (с ограничением в 1 символ это значение = 5)
                print('|  '+str(key)+' ', end="") # Создание ячейки со значением в ней
            else: # Эта ветка выполняется, если условие выше неверно
                print('|  '+str(key)+(' ')*(dlina_value-(len(str(key)))), end="") # Создание ячейки со значением в ней
    else: # Эта строка кода выполняется, если цикл не был прерван командой break
        print("|") # Печатаем последний символ во второй строке
    print('+-----+'+('-'*(dlina_value+2)+'+')*10+'-'*(dlina_value+2)+'+') # Печатаем третью строку таблицы
    for _, items in d.items(): 
        if items == "Y": # Условие - если ключ = Y, то выполняется дальнейший код программы
            print("| ",items," ",end="") # Создание ячейки со значением в ней Y
        else: # Эта ветка выполняется, если условие items = Y, неверно
            if len(str(items)) == 7: # Сравнивает длину ключа с фиксированным значением для корректроного отображения значения в ячейке. Фиксированное число зависит от округления значения (с ограничением в 4 символа это значение = 7)
                print('| '+str(items)+' ', end="") # Создание ячейки со значением в ней
            else:# Эта ветка выполняется, если условие выше неверно
                print('| '+str(items)+(' ')*(dlina_value-(len(str(items)))+1), end="") # Создание ячейки со значением в ней
    else: # Эта строка кода выполняется, если цикл не был прерван командой break
        print("|") # Печатаем последний символ в четвертой строке
    print('+-----+'+('-'*(dlina_value+2)+'+')*10+'-'*(dlina_value+2)+'+')# Печатаем пятую строку таблицы    
f()