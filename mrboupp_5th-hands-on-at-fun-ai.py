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
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import seaborn as sns
pokemon = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

pokemon.head()
pokemon.info()
pokemon.describe(include= 'O')
pokemon.describe()
pokemon.fillna("NoType",inplace=True)


pokemon_ability = pokemon.drop(columns=['Name','Type 1','Type 2'])

#df_n = df.select_dtypes(include=[np.number])これは数字のみ持ってくるやつ。型別で持ってくるメソッド
pokemon_ability.head()
tmp = pd.DataFrame(pokemon['Type 1'])
tmp
fig, ax = plt.subplots(figsize=(12,8))

sns.countplot(pokemon['Type 1'], ax=ax,order=pokemon['Type 1'].value_counts().sort_values().index)#.set_title('Type 1', fontsize=16)

plt.xlabel('Type 1', fontsize=15)

plt.show()
df = pd.DataFrame(pokemon['Type 2'])

df
fig, ax = plt.subplots(figsize = (12,8))

sns.countplot(pokemon['Type 2'],ax = ax,order = pokemon['Type 2'].value_counts().sort_values().index)

plt.show()
column = ['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric',

       'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice',

       'Dragon', 'Dark', 'Steel', 'Flying']
map_dis = df.join(tmp)
map_dis
#from collections import Counter

#Counter(map_dis)
map_dis.groupby(['Type 1','Type 2']).size().unstack().fillna(0).style.background_gradient(axis=1)
fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(15,40))



 

#plt.subplot(gs[0, :2])

sns.distplot(pokemon['Attack'][::2], color='green', ax=ax1,bins = 55).set_title('Attack', fontsize=16)

sns.distplot(pokemon['HP'][::2], color='green', ax=ax2,bins = 55).set_title('HP', fontsize=16)

sns.distplot(pokemon['Defense'][::2], color='green', ax=ax3,bins = 55).set_title('Defense', fontsize=16)

sns.distplot(pokemon['Sp. Atk'][::2], color='green', ax=ax4,bins = 55).set_title('Sp. Atk', fontsize=16)

sns.distplot(pokemon['Sp. Def'][::2], color='green', ax=ax5,bins = 55).set_title('Sp. Def', fontsize=16)

sns.distplot(pokemon['Speed'][::2], color='green', ax=ax6,bins = 55).set_title('Speed', fontsize=16)

sns.distplot(pokemon['Total'][::2], color='green', ax=ax7,bins = 100).set_title('Total', fontsize=16)





plt.show()
fig, ax = plt.subplots(figsize=(40,8))

sns.countplot(pokemon['Total'][::2], ax=ax).set_title('Total')

plt.show()





#gridkw = dict(height_ratios=[8,8,8,8,8,8])

fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(15,40))



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Total'], color='green',shade=True, ax=ax1).set_title('Total', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Total'],shade=True, ax=ax1).set_title('Legendasry Total', fontsize=16)



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'HP'], color='green',shade=True, ax=ax2).set_title('HP', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'HP'],shade=True, ax=ax2).set_title('Legendasry HP', fontsize=16)



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Attack'], color='green',shade=True, ax=ax3).set_title('Attack', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Attack'],shade=True, ax=ax3).set_title('Legendasry Attack', fontsize=16)



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Defense'], color='green',shade=True, ax=ax4).set_title('Defense', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Defense'],shade=True, ax=ax4).set_title('Legendasry Defense', fontsize=16)



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Sp. Atk'], color='green',shade=True, ax=ax5).set_title('Sp. Atk', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Sp. Atk'],shade=True, ax=ax5).set_title('Legendasry Sp. Atk', fontsize=16)



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Sp. Def'], color='green',shade=True, ax=ax6).set_title('Sp. Def', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Sp. Def'],shade=True, ax=ax6).set_title('Legendasry Sp. Def', fontsize=16)



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Speed'], color='green',shade=True, ax=ax7).set_title('Speed', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Speed'],shade=True, ax=ax7).set_title('Legendasry Speed', fontsize=16)





plt.show()
fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(15,80))



sns.boxplot(x="HP", y="Type 1", data=pokemon,ax=ax1, palette="Set3")

sns.boxplot(x="Attack", y="Type 1", data=pokemon,ax=ax2, palette="Set3")

sns.boxplot(x="Defense", y="Type 1", data=pokemon,ax=ax3, palette="Set3")

sns.boxplot(x="Sp. Atk", y="Type 1", data=pokemon,ax=ax4, palette="Set3")

sns.boxplot(x="Sp. Def", y="Type 1", data=pokemon,ax=ax5, palette="Set3")

sns.boxplot(x="Speed", y="Type 1", data=pokemon,ax=ax6, palette="Set3")

sns.boxplot(x="Total", y="Type 1", data=pokemon,ax=ax7, palette="Set3")



plt.show()
fig,ax = plt.subplots(1,figsize=(15,10))

sns.lineplot(x=pokemon['Generation'],y=pokemon['Total'])

plt.show()
columns = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']
#fig, (ax1, ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(15,80))

fig,(ax1) = plt.subplots(1,figsize=(15,10))

i=1



for column in columns:

    sns.lineplot(x=pokemon['Generation'],y=pokemon[column],ax = ax1)

    i=i+1

    

plt.show()
dff = pokemon.sort_values('HP', ascending=False).head(10)

dff
#sns.set()

#sns.set_style('whitegrid')

#sns.set_palette(sns.cubehelix_palette(24),n_colors=15)

# set the palette for 15 colors

current_palette = sns.cubehelix_palette(15, start = .3, reverse=True)

sns.set_palette("Blues_r", n_colors = 25)



#fig,ax = plt.subplots(1,figsize=(10,10))

#ax.barh(dff['Name'],dff['HP'])

#fig.show()

topworsd_plot = sns.barplot(dff['HP'],dff['Name'])

topworsd_plot.set(xlabel='Name',ylabel='HP')
current_palette = sns.cubehelix_palette(15, start = .3, reverse=True)

sns.set_palette("Blues_r", n_colors = 25)







dff = pokemon.sort_values('Attack', ascending=False).head(10)

topworsd_plot = sns.barplot(dff['Attack'],dff['Name'])

topworsd_plot.set(xlabel='Name',ylabel='Attack')





#import matplotlib.ticker as ticker

#for index, row in dff.iterrows():

#    topworsd_plot.text(row.name,row.Attack, round(row.Attack,2), color='black', ha="center")

#for p in dff.iterrows():

#    x=p.get_bbox().get_points()[:,0]

#    y=p.get_bbox().get_points()[1,1]

#    ax.annotate(p.get_height(),(x.mean(),y))
fig,ax = plt.subplots(1,figsize=(15,10))

sns.countplot(pokemon['Generation'])

plt.show()
fig,ax = plt.subplots(1,figsize=(15,10))

#sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Total'], color='green',shade=True, ax=ax1).set_title('Total', fontsize=16)

#plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)

list = [1,2,3,4,5,6]

for i in list:

    sns.countplot(pokemon.loc[(pokemon['Generation']==i),'Type 1'])

plt.show()
sns.set_palette("vlag", n_colors = 25)

fig ,ax = plt.subplots(1,figsize=(15,10))

#hue="Pclass"

sns.countplot(pokemon['Generation'],data=pokemon,hue="Legendary",ax=ax,palette='Set2')

#bar[2,4,6,8,10,12].set_color("blue")

plt.show()