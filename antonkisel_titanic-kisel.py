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
        
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import missingno as msno
import plotly.express as px

sns.set(style="darkgrid")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random
import matplotlib.pyplot as plt

# Прочитать CSV-файл и перевести его в DataFrame
df = pd.read_csv("/kaggle/input/titanic/train_and_test2.csv")
df.head()

# Прочитать CSV-файл (столбцы 'Name', 'Sex', 'Survived' из 10 первых строк) и перевести его в DataFrame
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=10, usecols=['Age', 'Sex', '2urvived'])
df
# Прочитать файл CSV и перевести каждую 100-ую строку в DataFrame
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', chunksize=100)
df_each = pd.DataFrame()
for chunk in df:
    df_each = df_each.append(chunk.iloc[0,:])
df_each
# Посмотреть информацию объекта DataFrame - формат столбцов, размернось, общую статистику
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=10)
 
# Формат каждого столбца
print('\n', 'Формат столбцов:')
print(df.dtypes)
 
# Размерность DataFrame
print('\n', 'Размерность:')
print(df.shape)
 
# Общая статистика
print('\n', 'Общая статистика')
print(df.describe())
# Проверить имеет ли df пропущенные значения
 
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=10)
 
# проверка на пропущенные значения
df.isnull().values.any()

# Получить наименование столбцов и сумму пропущенных значений в каждом

df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=20)
 
# получаем Series суммы пропущенных значений
 
# вариант 1
missing = df.isnull().sum()
 
# вариант 2
missing = df.apply(lambda x: x.isnull().sum())
 
print(missing[missing != 0])
# 47. Изменить позиции столбцов объекта DataFrame
 
# 1. Поменять местами столбцы 'a' и 'c'
# 2. Написать функцию, которая меняет столбцы местами
# 3. Сортировать столбцы по наименованию
 
df = pd.DataFrame(np.arange(20).reshape(4, 5), columns=list('abcde'))
 
# 1
df1 = df[list('cbade')]
print(df1)
print()
 
# 2
def switch_columns(df, col1=None, col2=None):
    
    if col1:
        if not col1 in df:
            return False
    if col2:
        if not col2 in df:
            return Fasle
    
    colnames = df.columns.tolist()
    i1, i2 = colnames.index(col1), colnames.index(col2)
    colnames[i2], colnames[i1] = colnames[i1], colnames[i2]
    return df[colnames]
 
df2 = switch_columns(df, 'd', 'a')
print(df2)
print()
 
# 3
df3 = df[sorted(df.columns)]
print(df3)
print()
# Создать новый столбец со значениями 0
 
# создаем DataFrame
df = pd.DataFrame(data=np.arange(20).reshape(4,5), columns=list('abcde'))
 
# добавляем столбец
df['f'] = 0
 
df
# Отобрать DataFrame по условию столбца "Age" (где отсутсвуют значения)
 
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=25)
df[df['Age'].isnull()]
# Отобрать DataFrame по нескольким условиям поля "Age" и "Sex" (пол - мужской, возраст - больше 30)
 
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=25)
df[(df['Sex'] == 'male') & (df['Age'] > 30)]
# Отобрать DataFrame по полю "Age" (между 30 и 40)
 
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=25)
df[df['Age'].between(30, 40)]
# Отобрать каждую 10-ую строку и столбцы ['Age', 'Name', 'Survived']
 
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv', nrows=100)
df.iloc[::10, :][['Age', 'Age', '2urvived']]
# Изменить данные столбца DataFrame по условию (по многим условиям)
 
df = pd.read_csv('/kaggle/input/titanic/train_and_test2.csv')
 
def change_values(val):
    
    # Проверяем на числовой тип входящей переменной
    try:
        float(val)
    except Exception as e:
        return val
    
    # Условия изменения значений
    if val > 25:
        return 'High'
    elif val < 25:
        return 'Low'
    
col_df = df['Age'].apply(change_values)
col_df