# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/datata"))



# Any results you write to the current directory are saved as output.
table1 = pd.read_csv("../input/datasetss/_3125350.csv")
#перевод float в str

# B = table3['Кадастровый номер']

# list = []

# for i in range(len(B)):

#      a = str(B[i])

#      list.append(a)

# table3['Кадастровый номер'] = list





#определяем типы данных и удаляем не числовые

list_types = [str(type(item)) for item in table1['Кадастровый номер']]

table_t = pd.DataFrame(list_types, columns = ['type'])

table_t['range'] = np.arange(len(table_t))

print (table_t['type'].value_counts())

#

uslovie = np.where(table_t['type'].str.contains("<class 'int'>"), 1, 0)

table_t['uslovie'] = uslovie

table_iskl = table_t[table_t['uslovie'] != 0]

list_of_i = table_iskl['range'].astype(int)



#удаляем не float значения

# table1 = table1.drop(table1.index[[list_of_i]])

table1['Кадастровый номер'][:900000]
table1.iloc[:5]
table2 = pd.read_excel("../input/datata/1.  2019 ---  .xlsx")
table3 = pd.DataFrame(table2, columns = ['Кадастровый номер1', 'Кадастровый номер помещение', 'raion', 'city',

       'city2', 'street', 'd', 'k', 'str', 'pom','Кадастровый номер'])
table3.columns
table_itog = pd.merge(table3,table1, on='Кадастровый номер', how = 'left')


write = pd.ExcelWriter('../input/itog.xlsx')

table_itog.to_excel(write)

write.close()
table_itog
a = pd.Series(table3['Кадастровый номер'])

table3['Кадастровый номер'] = a.astype(object)
table3
table_itog.columns