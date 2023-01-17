# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

df.tail()
df.drop('date', axis=1, inplace=True)

df.tail()
df.month.replace('Janeiro', 'January',inplace=True)

df.month.replace('Fevereiro','February',inplace=True)

df.month.replace('Março','March',inplace=True)

df.month.replace('Abril','April',inplace=True)

df.month.replace('Maio','May',inplace=True)

df.month.replace('Junho','June',inplace=True)

df.month.replace('Julho','July',inplace=True)

df.month.replace('Agosto','August',inplace=True)

df.month.replace('Setembro','September',inplace=True)

df.month.replace('Outubro','October',inplace=True)

df.month.replace('Novembro','November',inplace=True)

df.month.replace('Dezembro','December',inplace=True)

df.tail()
df['number'].describe()
print('Max number of registered fires in a month: ',df['number'].max())

print('State: ',df[df['number'] ==  df['number'].max()]['state'].iloc[0])

print('Year: ',df[df['number'] ==  df['number'].max()]['month'].iloc[0])

print('Month: ',df[df['number'] ==  df['number'].max()]['year'].iloc[0])
queim_sum_mês = df.groupby(['month'], as_index=False).sum() 

queim_sum_mês.drop('year',axis=1, inplace=True)



piores_meses = queim_sum_mês[queim_sum_mês['number']>queim_sum_mês['number'].mean()+queim_sum_mês['number'].std()]

print('Worst months: ')

for i in range(len(piores_meses)):

    print(piores_meses['month'].values[i])



sns.catplot(x='month', y='number', kind='bar',data=df[['month','number']], aspect=5, estimator=sum);
queim_sum_estados = df.groupby(['state'], as_index=False).sum()

queim_sum_estados.drop('year',axis=1, inplace=True)





sns.catplot(x='state', y='number',data=df[['state','number']], kind='bar', aspect=5, estimator=sum);



piores_estados = queim_sum_estados[queim_sum_estados['number']> queim_sum_estados['number'].mean() +queim_sum_estados['number'].std()]

print('Worst states: ')

for i in range(len(piores_estados)):

    print(piores_estados['state'].values[i])
queim_sum_ano = df.groupby(['year'], as_index=False).sum() 



piores_anos = queim_sum_ano[queim_sum_ano['number']>queim_sum_ano['number'].mean() + queim_sum_ano['number'].std()]

print('Worst years: ')

for i in range(len(piores_anos)):

    print(piores_anos['year'].values[i])





plt.figure(figsize=[12,7])

plt.xlim([1998, 2017])

plt.title('Registered number of fires per year (sum of all entries)')

sns.lineplot(x='year', y='number',data=queim_sum_ano);
poly = np.polyfit(queim_sum_ano['year'],queim_sum_ano['number'],3)

z = np.poly1d(poly)

    

anos = np.linspace(1998, 2017, 20)



plt.figure(figsize=[12,7])

plt.plot(anos, queim_sum_ano['number'], '-', label='Real data') 

plt.plot(anos,z(anos), '--', label='Fitted curve')

plt.xlim([1998, 2017])

plt.ylim([17000, 48000])

plt.title('Fitting the real data into a curve (all registered years)')

plt.legend()

plt.show()
for i in range(2019,2024,1):

    print(i, '->', math.trunc(z(i)))
new_model = queim_sum_ano[queim_sum_ano['year']>2006]

poly = np.polyfit(new_model['year'],new_model['number'],1)

z = np.poly1d(poly)

    

anos = np.linspace(2007, 2017, 11)



plt.figure(figsize=[12,7])

plt.plot(anos, new_model['number'], '-', label='Real data') 

plt.plot(anos,z(anos), '--', label='Fitted curve')

plt.xlim([2007, 2017])

plt.ylim([17000, 48000])

plt.title('Fitting the real data into a curve (years>2006)')

plt.legend()

plt.show()
for i in range(2019,2024,1):

    print(i, '->', math.trunc(z(i)))