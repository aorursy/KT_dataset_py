# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
m=np.random.randint(0,100,100) #генерация массива m со случайными числами от 0 до 100 размером в 100 элементов
m #вывод m
day=['Monday','Tuesday','Wednesday'] #объявление списка день с элементами понедельник, вторник, среда
day #вывод списка день
m1=np.random.randint(0,100,100) #генерация массива m1 со случайными числами от 0 до 100 размером в 100 элементов
m2=np.random.randint(0,100,100) #генерация массива m2 со случайными числами от 0 до 100 размером в 100 элементов
dic={'Monday':m,'Tuesday':m1,'Wednesday':m2} #создание словаря dic- дням соответствуют массивы
table=pd.DataFrame(dic) #создаем table как двумерную таблицу с элементами словаря dic
table # вывод table
plt.figure(figsize=(15,10)) #создание графика 15 на 10
table['Monday'].plot() #график понедельник
table['Tuesday'].plot() #график вторник
table['Wednesday'].plot() #график среда
x=[i for i in range(0,101,1)] #x изменяется от 0(вкл) до 101(не вкл) с шагом в 1
plt.figure(figsize=(15,10)) #создаем график 15 на 10
plt.plot(table['Monday'],'red') #график таблицы понедельник красным цветом
plt.scatter(x='Monday',y='Tuesday',data=table) #график рассеяния понедельника и вторника
plt.figure(figsize=(15,10)) #создаем график 15 на 10
plt.scatter(x='Monday',y='Tuesday',data=table) #график рассеяния понедельника и вторника
plt.scatter(x='Monday',y='Tuesday',data=table,marker='s') #график рассеяния понедельника и вторника с маркером s(квадрат) 
table['Monday'].plot.bar() #график понедельника полосами (столбцами)
plt.figure(figsize=(17,10)) #создаем график 17 на 10
plt.subplot(1,2,1) #подстрока в два ряда и один столбец, 1-й ряд
plt.plot(table['Monday']) #график понедельника
plt.subplot(1,2,2) #подстрока в два ряда и один столбец, 2-й ряд
plt.plot(table['Tuesday'],'red') #график вторника с красным маркером


