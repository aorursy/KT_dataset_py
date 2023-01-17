# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import  matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fire_df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv',encoding='cp437')

fire_df.head(10)
fire_df.isna().sum()
fire_df.info()
fire_df.describe()
month_fire = fire_df['month'].unique()

month_fire
fire_df['month'].replace(to_replace = 'Janeiro', value = 'Jan', inplace = True)

fire_df['month'].replace(to_replace = 'Fevereiro', value = 'Feb', inplace = True)

fire_df['month'].replace(to_replace = 'Março', value = 'Mar', inplace = True)

fire_df['month'].replace(to_replace = 'Abril', value = 'Apr', inplace = True)

fire_df['month'].replace(to_replace = 'Maio', value = 'May', inplace = True)

fire_df['month'].replace(to_replace = 'Junho', value = 'Jun', inplace = True)

fire_df['month'].replace(to_replace = 'Julho', value = 'Jul', inplace = True)

fire_df['month'].replace(to_replace = 'Agosto', value = 'Aug', inplace = True)

fire_df['month'].replace(to_replace = 'Setembro', value = 'Sep', inplace = True)

fire_df['month'].replace(to_replace = 'Outubro', value = 'Oct', inplace = True)

fire_df['month'].replace(to_replace = 'Novembro', value = 'Nov', inplace = True)

fire_df['month'].replace(to_replace = 'Dezembro', value = 'Dec', inplace = True)
states_fire = fire_df['state'].unique()

states_fire
%matplotlib inline

states_fire_df = fire_df[['state','number']]



fire_state = states_fire_df.groupby(['state']).sum().sort_values('number',ascending=False).reset_index()

fire_state.plot(x='state',y='number',kind='bar')

plt.show()

print(f'Maximun number of fire - \n{fire_state.head(1)}\n Minimum number of forest fire - \n{fire_state.tail(1)}')
fire_by_months = fire_df[['month','number']]

fire_by_months = fire_by_months.groupby('month').sum().sort_values('number',ascending=False).reset_index()

fire_by_months.plot(x='month',y='number',kind='bar')

plt.show()

print(f'Maximun number of fire - \n{fire_by_months.head(1)}\n Minimum number of forest fire - \n{fire_by_months.tail(1)}')
fire_by_year_df = fire_df[['number','year']]

fire_by_year = fire_by_year_df.groupby('year').sum().reset_index()

fire_by_year.plot(x='year',y='number')

plt.show()

print(f'Maximun number of fire - \n{fire_by_year.max()}\n\nMinimum number of forest fire - \n{fire_by_year.min()}')