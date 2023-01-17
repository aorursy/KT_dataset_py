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
with open('../input/metadata.txt') as f:

    print(f.read())
df = pd.read_csv('../input/south-korea-total-life-table-1970-2017-by-feature.csv')

df.keys()
df.head()
df.tail()
df['연령별'].values[0], df['시점'].values[0]
# Create new columns with age and year as integers

from itertools import groupby

df['age'] = [int([''.join(g) for _, g in groupby(value, str.isalpha)][0]) for value in df['연령별'].values]

df['year'] = [int([''.join(g) for _, g in groupby(value, str.isalpha)][0]) for value in df['시점'].values]



df.head()
my_year = 1986

print(df[(df.year==my_year) & (df.age==0)][['age','year','기대여명(남자)[년]']])

print(f'half of life expectancy at birth: {df[(df.year==my_year) & (df.age==0)]["기대여명(남자)[년]"].values[0] / 2}')

print(f'current age: {2019 - my_year}')
# see all rows that correspond to my age at a certain year

df[(df.year-df.age==my_year)]
df[(df.year-df.age==my_year) & (df.year==2017)][['age','year','기대여명(남자)[년]']]
my_df = df[(df.year-df.age==my_year)]

my_df['total life expectancy (male)'] = my_df['age'] + my_df['기대여명(남자)[년]']

my_df[['year','total life expectancy (male)']]
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))



x = my_df['age'] # 0 ~ 31

y = my_df['total life expectancy (male)']

plt.scatter(x,y, color="blue")



poly = np.polyfit(x, y, deg=2)



ages = [i for i in range(0,41)] # 0 ~ 40

y_extrapolated = [np.polyval(poly, age) for age in ages]

for age in ages:

    plt.plot(ages, y_extrapolated, color="red")



plt.show()
y_extrapolated[33]
y_extrapolated[40]
print('current age, total life expectancy, remaining life expectancy (percentage)')

for i in range(100):

    life_exp = np.polyval(poly, i)

    print(f'{i}, {life_exp:.2f}, {(100 * (life_exp-i) / life_exp):.2f}%')