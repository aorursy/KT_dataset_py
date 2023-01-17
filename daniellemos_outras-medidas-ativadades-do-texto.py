import pandas as pd

import numpy as np

import matplotlib as plt

import math
male = np.array([4.5, 6.1, 3.2, 6.9, 7.1, 8.2, 3.3, 2.5, 5.6, 7.2, 3.4])

female = np.array([6.3,  6.8,  5.9,  6.0,  4.9,  6.1,  6.3,  7.5,  7.7,  6.5])

df = pd.DataFrame(index = ["Homens", "Mulheres"], data = [male, female])

df
turma = np.concatenate((male, female))

turma.mean()
turma.var() ** (1/2)
df.T.corr()
df2 = pd.DataFrame({'UF':'RO AC AM RR PA AP TO MA PI CE RN PB PE AL SE BA MG ES RJ SP PR SC RS MS MT GO DF'.split(),

             'Densidade Populacional (hab/km2)':[6, 4, 2, 2, 5, 4, 5, 17, 12, 51, 53, 61, 81, 102, 81, 24, 31, 68, 328, 149, 48, 57, 37, 6, 3, 15, 353],})

df2
med = df2['Densidade Populacional (hab/km2)'].mean()

scorelist = []

for i in df2['Densidade Populacional (hab/km2)']:

    scorelist.append((i-med)/df2['Densidade Populacional (hab/km2)'].var()**(1/2))

dfscore = pd.DataFrame({'UF': df2['UF'].values, 

              'Escore Padronizado': scorelist})

dfscore
valor_med = dfscore['Escore Padronizado'][dfscore['Escore Padronizado'] > 1].sum()/3

valor_med
dfscore[dfscore['Escore Padronizado'] > valor_med]
tabel = pd.DataFrame(index = range(2,10), data = [[9.],[7, 8],[7, 9],[2, 6, 8],

                                          [0, 2, 3, 3, 3, 5, 5, 6, 8, 8, 9, 9],

                                          [0, 0, 1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 9],

                                          [1, 1, 2, 2, 3, 3, 4, 5, 7, 7, 8, 9],

                                          [0, 1, 4, 7]])

tabel
moda = tabel.mode(axis = 1).loc[6][0]

moda
media = tabel[0].sum()/8

media
desv_padrao = tabel[0].var()**(1/2)

desv_padrao
(media - moda)/desv_padrao
sal = pd.Series([6300, 5700, 4500, 3800, 3200, 7300, 7100, 5600, 6400, 7000, 3700, 6500, 4000,

5100, 4500], name='Salário')

sal
q1 = sal[pd.Series(sal[sal.index[sal.index < pd.Series(sal.index).median()]].index).median()]

q1
q3 = sal[pd.Series(sal[sal.index[sal.index > pd.Series(sal.index).median()]].index).median()]

q3
q2 = sal.median()

q2
((q3 - q2) - (q2 - q1)) / ((q3 - q2) + (q2 - q1))
sal = pd.DataFrame([6300, 5700, 4500, 3800, 3200, 7300, 7100, 5600, 6400, 7000, 3700, 6500, 4000,

5100, 4500], columns = ['Salário'])

sal.boxplot(column = 'Salário')