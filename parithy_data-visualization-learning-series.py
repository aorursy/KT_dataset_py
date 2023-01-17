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
import matplotlib.pyplot as plt
import seaborn as sns
pokemon=pd.read_csv("../input/pokemon.csv")
pokemon.head()
pokemon.columns
pokemon['type1'].value_counts().plot.bar()
pokemon['hp'].value_counts().sort_index().plot.line()
pokemon['weight_kg'].plot.hist()
pokemon.plot.scatter(x='attack',y='defense',figsize=(12,6),title='Attach va Defense comparisons')
pokemon.plot.hexbin(x='attack',y='defense',gridsize=15)
pokemon_stats_legendary = pokemon.groupby(['is_legendary', 'generation']).mean()[['attack', 'defense']]
pokemon_stats_legendary.plot.bar(stacked=True)
pokemon_stats_by_generation = pokemon.groupby('generation').mean()[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
pokemon_stats_by_generation.plot.line()
ax1 = pokemon['base_total'].plot.hist(figsize=(12,10),bins=50,color='green',fontsize=14)
ax1.set_title("Pokemon Total By Stats",fontsize=10)
ax = pokemon['type1'].value_counts().plot.bar(
    figsize=(12, 9),
    fontsize=14
)
ax.set_title("Pokemon Type By Count",fontsize=25)
sns.despine(bottom=True,left=True)
fig ,arry = plt.subplots(2,1,figsize=(8,8))
pokemon['defense'].plot.hist(ax=arry[0],title='Pokemon Attack Ratings',color='mediumvioletred')
pokemon['attack'].plot.hist(ax=arry[1],title='Pokemon Defense Ratings',color='green')
sns.despine(left=True)
sns.countplot(pokemon['generation'])
sns.distplot(pokemon['hp'].dropna())
#sns.jointplot(x='attack',y='defense',data=pokemon)
sns.jointplot(x=pokemon['attack'],y=pokemon['defense'],color='orange')
sns.jointplot(x='attack',y='defense',data=pokemon,kind='hex',color='violet')
sns.boxplot(pokemon['is_legendary'], pokemon['attack'])
sns.kdeplot(pokemon['hp'], pokemon['attack'])
sns.violinplot(pokemon['is_legendary'], pokemon['attack'])
g=sns.FacetGrid(pokemon,row="is_legendary")
g.map(sns.kdeplot,'attack')
g=sns.FacetGrid(pokemon,col="is_legendary",row='generation')
g.map(sns.kdeplot,'attack',color='darkgreen')
sns.pairplot(pokemon[['hp','attack','defense']])
sns.lmplot(x='attack',y='defense',hue='is_legendary',markers=['*','o'],fit_reg=False, data=pokemon)
sns.heatmap(pokemon.loc[:,['hp','attack','defense','speed']].corr(),annot=False)
sns.heatmap(pokemon.loc[:,['hp','attack','defense','speed']].corr(),annot=True)