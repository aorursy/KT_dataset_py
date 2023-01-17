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
#Первый вопрос
#import pandas as pd
print(pd.__version__)
# Второй вопрос
A='Series'
B="DataFrame"
print("Для работы с данными в pandas используется тип",A,"для обработки одномерных массивов и тип",B,"для обработки таблиц.")
#Третий вопрос
print("float64")
print("datetime64[ns]")
print("float32")
print("int32")
print("category")
print("object")
#Четвертый вопрос
s = pd.Series(range(1,11))
display(s)
ss = pd.Series(np.array([i+1 for i in range(10)]))
display(ss)
#Пятый вопрос
df=pd.DataFrame(np.random.rand(10,3),columns=["A","B","C"])
display(df)
#Шестой вопрос
df.tail(5)
#Седьмой вопрос
print(df.columns)
#Восьмой вопрос
df1=pd.DataFrame(np.random.rand(10,3),columns=["A","B","Result"])
display(df1)
df1.sort_values("Result",ascending=0)
#Девятый вопрос
a='loc'
b='iloc'
print("Для получения элементов DataFrame по названиям столбцов и индексов используется команда",a,", а по номерам столбцов и индексов - команда",b)
#Десятый вопрос
print('drop')