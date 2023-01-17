# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#1



print(pd.__version__)
#2

print("Для работы с данными в pandas используется тип Series, для обработки одномерных массивов и тип DataFrame для обработки таблиц")

#3

print("типы данных, используемые в pandas:","float64","datetime64[ns]","float32","int32","category","object",sep='\n')
#4

print("Выберите правильные способы создания одномерного массива данных с числами от 1 до 10.")



a=pd.Series(range(1,11))

display(a)

b=pd.Series(np.array([i+1 for i in range(10)]))

display(b)
#5

print("Какая команда создаст структуру DataFrame из 10 строк и 3 столбцов, заполненных случайными значениями от 0 до 1, названия столбцов - латинские буквы.")



c=pd.DataFrame(np.random.rand(10,3),columns=["A","B","C"])

display(c)
#6

print("Как посмотреть только 5 нижних строк DataFrame?")



c.tail(5)
#7

print("Какой командой в DataFrame с именем df можно получить перечень названий столбцов")



print(c.columns)
#8

print("Как в DataFrame с именем df отсортировать данные по столбцу 'Result' по убыванию?")



df=pd.DataFrame(np.random.rand(10,3),columns=["A","B","Result"])

df.sort_values("Result",ascending=0)
#9

print("Для получения элементов DataFrame по названиям столбцов и индексов используется команда 'loc', а по номерам столбцов и индексов - команда 'iloc'.")
#10

print("Для удаления элементов DataFrame ипользуется команда 'drop'")