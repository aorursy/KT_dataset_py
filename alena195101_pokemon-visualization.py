# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image

from IPython.core.display import HTML



import matplotlib.pyplot as plt



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

data.head()
from IPython.display import display, HTML

s=""

for i in range(43):

    s+='<img src="{}" style="display:inline;margin:1px; width:150px"/>'.format("https://pokemongolife.ru/p/"+data['Name'][i]+".png",

                                                                                    data['Name'][i])



display(HTML(s))

data_stat = pd.DataFrame()

data_stat['min'] = data.iloc[:,4:-2].min()

data_stat['max'] = data.iloc[:,4:-2].max()

data_stat['mean'] = data.iloc[:,4:-2].mean()

data_stat['median'] = data.iloc[:,4:-2].median()

data_stat['std'] = data.iloc[:,4:-2].std()

data_stat
print("Количество покемонов с 2мя типами: ", data[(~data['Type 1'].isna())&(~data['Type 2'].isna())].shape[0])

print("Количество покемонов с одним типом: ", data[(~data['Type 1'].isna())&(data['Type 2'].isna())].shape[0])
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(8,6))

data['Generation'].hist()



plt.xlabel('Поколение')

plt.title("Количество покемонов в поколениях")

ax.grid(axis = 'x')
type_df = data['Type 1'].value_counts().reset_index().merge(data['Type 2'].value_counts().reset_index())

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(10,8))



ax.barh(np.arange(len(type_df)), type_df['Type 1'], alpha = 0.5, color='y',label = 'Type 1')

ax.set_yticks(np.arange(len(type_df)))

ax.set_yticklabels(type_df['index'])



ax.barh(np.arange(len(type_df)), type_df['Type 2'], alpha = 0.5, color='g', label = 'Type 2')

ax.set_yticks(np.arange(len(type_df)))

ax.set_yticklabels(type_df['index'])



plt.xticks(range(0,150,10))



ax.invert_yaxis()  # labels read top-to-bottom



ax.grid(axis = 'y')

ax.set_title('Статистика по типам')

plt.legend()

plt.show()
data_stat = pd.melt(data, id_vars=['#',"Name", "Type 1", "Type 2",'Generation','Legendary'], var_name="Stat")

total_stat =data_stat[data_stat['Stat']=='Total']



data_stat = data_stat[data_stat['Stat']!='Total']
sns.set(style="whitegrid")



fig, (ax1, ax2) = plt.subplots(

                                nrows=1, ncols=2,figsize=(30, 10)

                                )



sns.boxplot(x="Stat", y="value", data=data_stat, hue="Legendary",palette="Set3", ax=ax1)

sns.boxplot(x="Stat", y = 'value', data=total_stat, hue="Legendary",palette="Set3", ax= ax2)

plt.show()
sns.set(style="whitegrid")



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(22,8))

sns.boxplot(x="Stat", y="value", data=data_stat, hue="Generation",palette="Set3", ax=ax1)

sns.boxplot(x="Stat", y = 'value', data=total_stat, hue="Generation",palette="Set3", ax= ax2)

plt.show()
d = data.groupby(['Generation','Legendary']).agg({'Attack':'mean', 'Defense':'mean'}).reset_index().sort_values(by=['Legendary'])

d['Generation'] = d['Generation'].astype(str)

d['Legendary'] = d['Legendary'].astype(str)

d['leg_gen'] = d['Legendary'].str.cat(d['Generation'], sep =" ")



sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(10,8))



ax.barh(np.arange(d.shape[0]), d['Attack'], alpha = 0.5, color='y',label = 'Attack')

ax.set_yticks(np.arange(d.shape[0]))

ax.set_yticklabels(d['leg_gen'])



ax.barh(np.arange(d.shape[0]), d['Defense'], alpha = 0.5, color='g',label = 'Defense')

ax.set_yticks(np.arange(d.shape[0]))

ax.set_yticklabels(d['leg_gen'])



ax.grid(axis = 'y')

plt.legend()

plt.show()
#sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(25,8))



d = data_stat.groupby(['Type 1','Stat']).value.mean().reset_index()

sns.barplot(x='Type 1',y='value',data = d, hue='Stat', palette = 'pastel')



plt.title('Стат/Тип')

plt.show()
fig, ax = plt.subplots(figsize=(25,8))



d_type2 = data[~data['Type 2'].isna()].copy()

d_type2['type'] = d_type2['Type 1'].str.cat(d_type2['Type 2'], sep =" ")

d_type2['type'] = d_type2['type'].apply(lambda x: x if x in d_type2['type'].value_counts()[d_type2['type'].value_counts()>6].index else None)

d_type2 = d_type2[~d_type2['type'].isna()]



data_stat2 = pd.melt(d_type2, id_vars=['#',"Name", "Type 1", "Type 2",'type','Generation','Legendary'], var_name="Stat")

total_stat2 = data_stat2[data_stat2['Stat']=='Total']

data_stat2=data_stat2[data_stat2['Stat']!='Total']

data_stat2 = data_stat2.groupby(['type','Stat']).value.mean().reset_index()



sns.barplot(x='type',y='value',data = data_stat2, hue='Stat', palette = 'pastel')



plt.title('Стат/Тип')

plt.show()

d_type2 = data[~data['Type 2'].isna()].copy()

d_type2['type'] = d_type2['Type 1'].str.cat(d_type2['Type 2'], sep =" ")

d_type2['type'] = d_type2['type'].apply(lambda x: x if x in d_type2['type'].value_counts()[d_type2['type'].value_counts()>6].index else None)

d_type2 = d_type2[~d_type2['type'].isna()]



data_stat2 = d_type2.groupby('type').mean()[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

data_stat2





fig, ax = plt.subplots(3, 3, subplot_kw=dict(projection='polar'),figsize = (20,20))

#fig.add_subplot(111, polar=True)

ind = list(data_stat2.index)

k=0

for i in range(3):

    for j in range(3):

        stats=data_stat2.loc[ind[k],:].values



        angles=np.linspace(0, 2*np.pi, data_stat2.shape[1], endpoint=False)

        # close the plot

        stats=np.concatenate((stats,[stats[0]]))

        angles=np.concatenate((angles,[angles[0]]))

        ax[i][j].plot(angles, stats, 'o-', linewidth=2)

        ax[i][j].fill(angles, stats, alpha=0.25)

        ax[i][j].set_thetagrids(angles * 180/np.pi, data_stat2.columns)

        ax[i][j].set_title(ind[k])

        ax[i][j].grid(True)

        k+=1

sns.jointplot(x='Attack', y='Defense', data=data)                          

plt.show()
sns.jointplot(x='Sp. Atk', y='Sp. Def', data=data) 

plt.show()
sns.jointplot(x='HP', y='Attack', data=data) 

plt.show()