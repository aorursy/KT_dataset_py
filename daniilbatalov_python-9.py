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
d1 = { "Россия":"Russia","Авторучка":"Pen","Инжир":"Fig","Пулемет":"Machinegun","Перец":"Pepper","Вертолет":"Helicopter","Карандаш":"Pencil","Нож":"Knife","Ботинок":"Boot","Поле":"Field" } # Вручную создали словарь из 10 ключей и их значений
print(d1) # Вывод на экран словаря
print("Задание №2")
print(d1[input("Какое слово нужно найти?: ")]) # Вывод английского слова по введенному русскому
print("Задание №3")
word_A = input("Какое слово нужно найти?: ") # Считывание слова с клавиатуры
for key, item in d1.items(): # Цикл, проходящий по всем ключам/значениям словаря
    if word_A == item: # Если слово пользователя есть в словаре в качестве значения, то выполняется дальнейший код по условию
        print(key) # Вывод на экран русского слова
print("Задание №4") # Программный код, призванный заменить код из Задания №2 и Задания №3
word_R = input("Какое слово(писать по-русски) нужно найти?: ") # Считывание слова с клавиатуры
print(d1[word_R]) if (word_R in d1) else print("Слово отсутствует в словаре!") # Вывод английского слова по введенному русскому
word_A = input("Какое слово(писать по-английски) нужно найти?: ") # Считывание слова с клавиатуры
for key, items in d1.items(): # Цикл, проходящий по всем ключам/значениям словаря
        if word_A == items: # Если слово пользователя есть в словаре в качестве значения, то выполняется дальнейший код по условию
            print(key) # Вывод на экран русского слова
            break # Прерывание цикла
else: # Коэ этой ветки выполняется, если выход из цикла был естественным (не использую конструкцию break)
    print("Слово отсутствует в словаре!") # Вывод сообщения на экран