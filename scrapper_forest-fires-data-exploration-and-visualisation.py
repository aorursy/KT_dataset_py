import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1' )

data.head()
data.duplicated().value_counts()
data.drop_duplicates(inplace=True)

data.shape
data['month'] = pd.Categorical(data.month,ordered=True,categories= ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho',

       'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'])
top_values = data.groupby('state')['number'].agg([sum,np.mean]).sort_values(by = 'sum',ascending = False)

top_values.head(10)
plt.figure(figsize=(10,8))

plt.title('Total values of Fire in states(total years)')

sns.set_style('white')

sns.barplot(y =top_values.index, x='sum', data=top_values, palette='Set2')
top_mean_values = top_values.sort_values(by = 'mean',ascending = False)

top_mean_values.head()
plt.figure(figsize=(7,5))

plt.title('States with Most Avg value of Fire (total years)')

sns.barplot(x ='mean',y =top_mean_values.index[:10], data=top_mean_values.iloc[:10,:], palette='Set2')
year_wise = data.groupby(['state','year'], as_index=False)['number'].agg([sum,np.mean])

year_wise.reset_index(level='state', inplace =True)

year_wise.head()
plt.figure(figsize=(14,4))

plt.ylim(0,2000)

plt.title('varaitaion of fire with year')

sns.barplot(y ='sum', x=year_wise.index,data =year_wise)
plt.figure(figsize=(16,4))

sns.set_style('whitegrid')

ax1 = plt.subplot(1,2,1)

plt.title('varaitaion of fire with year')

sns.lineplot(y ='sum', x=year_wise.index,data =year_wise)

plt.xlim(1997,2020)

plt.subplot(1,2,2,sharex =ax1)

plt.title('Avg varaitaion of fire with year')

sns.lineplot(y ='mean', x=year_wise.index,data =year_wise,)
year_wise_top = year_wise[year_wise['state'].isin(['Mato Grosso', 'Paraiba', 'Sao Paulo', 'Rio', 'Bahia'])]

year_wise_top.head()
plt.figure(figsize=(12,6))

sns.set_style('whitegrid')

plt.xlim(1997,2020)

plt.title('varaitaion of fire with year')

sns.lineplot(y ='sum', x=year_wise_top.index,data =year_wise_top, hue='state')
month_wise = data.groupby(['month','state'], sort = False)['number'].agg([sum,np.mean])

month_wise.reset_index(level='state', inplace =True)

month_wise.head(12)
plt.figure(figsize=(16,8))

ax1 = plt.subplot(2,1,1)

sns.set_style('white')

plt.ylim(0,4500)

sns.barplot(y ='sum', x=month_wise.index,data =month_wise,)

plt.subplot(2,1,2,sharex =ax1)

sns.lineplot(y ='mean', x=month_wise.index,data =month_wise,)