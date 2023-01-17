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
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',  encoding='ISO-8859-1')
''' How much empty columns do we have'''

df.isna().sum()
''' Unique states that has forest fire'''

states = df.state.unique()

print(f"States: {states}")

print(f"Numb: {len(states)}")
''' Get the states with more fire '''

top_10_states = df.groupby(['state'])['number'].sum().sort_values(ascending=False).head(10).keys().tolist()

top_10_values = df.groupby(['state'])['number'].sum().sort_values(ascending=False).head(10)
print(f"TOP 10 STATE \t VALUE")

print(top_10_values)
import matplotlib.pyplot as plt

import seaborn as sns
all_years_data = df.groupby('year').sum()

fig, ax = plt.subplots(figsize=(8,8))

plot = sns.lineplot(data=all_years_data, markers=True)

plot.set_title("Fire per year in Brazil")

plot.set(xlabel='Year', ylabel='No. of Fires')
fig, ax = plt.subplots(figsize=(15, 15))

plot = sns.barplot(x=top_10_states, y= top_10_values, palette='Accent')

plot.set(xlabel= 'State', ylabel='Number of fire occurence', title='Top 10 States with most fire occurence ')
''' Lets explore when the fire occour more on Mato Grosso '''

mato_grosso= df[df.state == 'Mato Grosso']

fire_sum_per_month = mato_grosso.groupby(['month'])['number'].sum().sort_values(ascending=False)

month = mato_grosso.groupby(['month'])['number'].sum().sort_values(ascending=False).keys().tolist()
''' Exploring mato grosso data by month'''

explode = len(month) * [0]

explode[0] = 0.1  ##Get the biggest falue and explode it 

fig, ax = plt.subplots( figsize= (8,8))

ax.pie(fire_sum_per_month, labels=month, autopct='%1.1f%%', startangle=145, explode=explode)

plt.tight_layout()

plt.show()
all_years_data = mato_grosso.groupby('year').sum()

fig, ax = plt.subplots(figsize=(8,8))

plot = sns.lineplot(data=all_years_data, markers=True)

plot.set_title("Fire per year in Mato Grosso")

plot.set(xlabel='Year', ylabel='No. of Fires')
setember = mato_grosso[mato_grosso.month == 'Setembro']

fig, ax = plt.subplots(figsize=(8,8))

plot = sns.lineplot(x='year', y='number', data=setember, markers=True)

plot.set_title("Setember fires over the Years in Mato Grosso")

plot.set(xlabel='YEAR', ylabel='No. of Fires')

print(f"Max: {mato_grosso.number.max()}")

print(f"Min: {mato_grosso.number.min()}")

print(f"Mean: {mato_grosso.number.mean()}")

plt.show()