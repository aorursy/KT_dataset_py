# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/vodafone-subset/vodafone-subset-6.csv', index_col='user_hash')

# Параметр index_col используем для того, чтобы взять в качестве индекса один из существующих в таблице столбцов.

# Наиболее часто это некие уникальные имена, id либо отметки даты/времени.
# Первые 10 строк

df.head(10)
# Последние 10 строк

df.tail(10)
# Размерность

df.shape
# кол-во строк

print(df.shape[0])

# кол-во столбцов

print(df.shape[1])
# Индексы

df.index
# Колонки

df.columns
# Типы колонок

df.dtypes
# Сводная информация

df.info()
# Весь столбец 

df['software_os_name']
# Конкретное значение в столбце

df['software_os_name']['ecfd365b8e9c6962b22e3f960c0616cc']
# Значения в столбце как массив Numpy

x = df['software_os_name'].values

# Обращение обычным способом, как срез по индексам массива

print(x[5:10])
# loc, iloc - особые формы обращения к значениям именно в датафреймах

print(df.iloc[4, 0])

print(df.loc['ecfd365b8e9c6962b22e3f960c0616cc', 'software_os_name'])
# slices

df_1 = df.iloc[:, :116]

df_1.head()
# Срез можно делать по последовательности индексов, даже если они не числовые

df_2 = df.loc['ab5aa49408d0b62b459ff835e8db433d':'ddb288b5480cd5ecd2f9dd884d07a82f', :]

df_2
# Срез по последовательности имён столбцов

df_3 = df.loc[:, 'calls_count_in_weekdays':'DATA_VOLUME_WEEKENDS']

df_3.head()
data_volume = df_3.iloc[:, -2:]

data_volume.head()
# Сравним распределения в двух столбцах. Верно ли, что трафик больше в будние дни?

data_volume.plot(kind='box')

plt.show()
# Фильтр по двум условиям

active_users = data_volume[ (df['DATA_VOLUME_WEEKDAYS'] != 0) | (df['DATA_VOLUME_WEEKENDS'] != 0) ]

active_users.head()
# Сколько осталось абонентов?

active_users.shape[0]
# Сравним распределения в двух столбцах у активных абонентов

active_users.plot(kind='box')

plt.show()
# Сравним по медиане

active_users['DATA_VOLUME_WEEKDAYS'].median(), active_users['DATA_VOLUME_WEEKENDS'].median()
# Функции типа median(), sum(), max() и т. п. можно применять ко всему датафрейму.

# По умолчанию функции применяются к столбцам.

active_users.median()
# Построим медианы в виде bar chart

active_users.median().plot(kind='bar')

plt.show()
# Гистограммы на двух разных картинках: параметр subplots

active_users.plot(kind='hist', subplots=True)

plt.show()
# Выбрали первые 8 столбцов - это данные по звонкам

calls = df_3.iloc[:, :8]

calls.head()
# Полезный метод для числовых колонок - показывает статистику

calls.describe()
# Есть ли вообще неактивные пользователи? Сколько таких?

(df_3.sum(axis=1) == 0).sum()
# Выбор колонок по именам, начинающимся с "calls_"

df_3[ df_3.columns[df_3.columns.str.startswith('calls_')] ]