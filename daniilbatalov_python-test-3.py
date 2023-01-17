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
# Напишите программу, которая вводит строку из нескольких слов
# разделенных пробелами, и создает список входящих в нее слов (используйте метод split)
# Вводимая строка берется из текстового файла.
def count(line): # Функция для подсчета количества уникальных слов в списке
    """
    Функция принимает в качестве параметра список.
    Функция подсчитывает количество уникальных элементов в списке
    и выводит на экран элемент и количество его повторений.
    """
    d = dict() # Создаем пустой словарь
    for i in line: # Перебираем все элементы в списке
        d[i]=line.count(i) # Запись в словарь. Ключ - слово, значение - количество его повторов
    print("Слова и их количество в прочитанном файле:") # Вывод сообщения на экран
    for key, value in d.items(): # Перебираем ключи и значения словаря d
        print(key, value) # Выводим на экран ключ и его значение
def unique_word(line): # Функция поиска уникальных элементов в списке
    """
    Функция принимает в качестве параметра список.
    Функция находит уникальные элементы в списке
    и выводит на экран по одному экземпляру каждого уникального элемента списка.
    """
    good_line = [] # Создаем пустой вспомогательный список
    for i in line: # Перебираем элементы списка
        if i not in good_line: # Если вспомогательный список не содержит в себе значение, то выполняется дальнейший код по условию
            good_line.append(i) # Добавление элемента во вспомогательный список
    print("Уникальные слова в файле:") # Вывод сообщения на экран
    print(good_line) # Вывод вспомогательного списка на экран
f = open('/kaggle/input/pythontest3/Python-test-3.txt','r', encoding='utf-8-sig') # Открываем файл на чтение и задаем кодировку текста (для корректного чтения)
line = f.read().split() # Читаем весь файл целиком, разделяя слова и преобразовывая их в список
f.close() # Закрываем файл
print(' '.join(line)) # Вывод на экран текста из файла
print() # Пустая строка
count(line) # Вызов функции count с параметром = прочитанный файл
print() # Пустая строка
unique_word(line) # Вызов функции unique_word с параметром = прочитанный файл