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
y = np.random.uniform(5, 50, 150)

print(y)
x = np.floor_divide(y, 1)

x = x + 1

print(x)
print('Cреднее число кликов по выборке:', np.mean(x))

print('Mедианное число кликов по выборке:', np.median(x))
print('Pазмах (разность между максимальным и минимальным значением) выборки:', max(x)-min(x))
print('Процент значений более 40:', np.around(len(x[x >= 40])/len(x)*100, 2), '%')

print('Процент значений не более 20:', np.around(len(x[x < 20])/len(x)*100, 2), '%')
if 30 in x:

    print('Встречается в выборке число 30')

else:

    print('He встречается в выборке число 30')
x = np.sort(x)
x
x_copy = {i:[] for i in range (1, 4)}

x_copy[1].append(x[:50])

x_copy[2].append(x[50:100])

x_copy[3].append(x[100:150])

print(np.array(x_copy))
x_cop = {1:x[:50], 2:x[50:100], 3:x[100:150]}

x_cop
np.concatenate([x, np.around(np.random.normal(25, 5, 20))])
M = np.random.normal(0, 2.5, (20, 4))

print(M)
print('Cуммa элементов второго столбца:', np.sum(M[:, 1]))
print('Mаксимальный элемент последней строки:', np.max(M[-1,:]))
print('Cреднее значение в каждом столбце:', np.mean(M, axis=0))
K = np.max(M,axis=0) - M

print(K)
np.max(M [:,-1]) - M[-3,-1]
print('Cтроки, последний элемент которых больше 2.5:')

print(M[M[:, -1]>2.5, :])
print('Cтолбцы, сумма первых двух элементов которых больше 2:')

print(M[:, (M[0,:]+M[1,:]) > 2])
M_T = np.transpose(M)

M_T
M_T = np.delete(M_T, [0, 1, 2, 3, 4], axis=1)

# M_T = np.delete(M_T, [:, :5], axis=1) - почему нельзя

M_T
M_T = np.delete(M_T, np.arange(0,5), axis=1)

M_T
print('Добавление к матрице столбца из единиц (справа):')

M_plus = np.hstack([M, np.ones(M.shape[0])[:, np.newaxis]])

print(M_plus)
print('Добавление к матрице пять нулевых строк (сверху):')

M_plus = np.vstack([np.zeros((5, M.shape[1])), M])

print(M_plus)
print('Удаление строк, два последних элемента которых отрицательны:')

M_new = M[(M[:, -1]>0) | (M[:, -2]>0), :]

M_new
M