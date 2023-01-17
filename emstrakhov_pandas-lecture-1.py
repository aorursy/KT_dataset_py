import numpy as np

import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0])

print(data)
data.values # массив значений
data.index # массив индексов
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

data
data.index
data['b']
data['a':'c']
data = pd.Series([0.25, 0.5, 0.75, 1.0], 

                 index=[2, 5, 3, 7])

data
data[2:3]
population_dict = {'California': 38332521,

                   'Texas': 26448193,

                   'New York': 19651127,

                   'Florida': 19552860,

                   'Illinois': 12882135}

population = pd.Series(population_dict)

population
population['Texas':'Illinois']
population[-3:] # последние три штата
population['Texas':'Illinois'].values
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,

             'Florida': 170312, 'Illinois': 149995}

area = pd.Series(area_dict)

area
states = pd.DataFrame({'population': population,

                       'area': area})

states
print(states)
states['population']
states['California'] # !!!
states.index
states.columns
states.values
a = np.random.normal(size=4).reshape((2, 2)) # матрица 2х2

df = pd.DataFrame(a, index=['row 1', 'row 2'], columns=['col 1', 'col 2'])

df
df2 = pd.DataFrame(a)

df2
states.columns['population'] = 'pop'
states.columns = ['pop', 'area']

data = states

data['density'] = data['pop'] / data['area']

data
data[data['density'] > 100]
data.iloc[1]
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])

ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])

pd.concat([ser1, ser2])
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])

ser2 = pd.Series(['D', 'E', 'F'], index=[3, 5, 6])

s = pd.concat([ser1, ser2])
s
s[3]
s.loc[3]
df1 = pd.DataFrame(np.arange(4).reshape((2, 2)), columns=['A', 'B'])

df1
df2 = pd.DataFrame(np.arange(5, 9).reshape((2, 2)), columns=['C', 'D'])

df2
df1.append(df2)
df1.append(df2, verify_integrity=True)