import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
pokemon = pd.read_csv('../input/Pokemon.csv',index_col='#')
pokemon.head(10)
#how many pokemons are added per generation?
gen = pokemon.Generation.value_counts().sort_index()
increment_gen = pd.Series([sum(gen[0:n]) for n in range(1,7)])
increment_gen.plot.line(title='Total Pokemons incremented by Generation',figsize=(8,4))

#how do the parameters evolve per generation on average?
plot_stat_gen = pokemon.groupby('Generation')['Attack','Defense','Sp. Atk','Sp. Def', 'Speed']\
.mean().plot.bar(figsize=(10,5), title='Avg stat per generation')
plot_stat_gen.legend(loc=4)
#strongest pokemon of each generation (and its type)
pkm = pokemon.reset_index()
pkm.head()
idx_strongest = pkm.groupby('Generation')['Total'].idxmax()
pkm.loc[idx_strongest,['#','Name','Type 1','Type 2']]
pokemon['Type 1'].unique()#types
#quantity of pokemons for each element
pokemon['Type 1'].value_counts().sort_index().plot.bar(figsize=(10,5), title='Number of Pokemon per Type 1')
sns.pairplot(pokemon.loc[:,'HP':'Sp. Def'])
#how do attack and defence correlate?
pokemon.plot.hexbin(x='Defense', y='Attack', gridsize=15, title='Attack vs Defence',figsize=(10,5))
#atk+sp. atk vs def + sp. def
joint_atk = pd.Series(map(lambda atk, sp_atk : atk+sp_atk,pokemon['Attack'], 
                          pokemon['Sp. Atk']), name='joint Attack stats')
joint_defense = pd.Series(map(lambda defense, sp_def : defense+sp_def, 
                              pokemon['Defense'], pokemon['Sp. Def']),
                          name='joint Defense stats')
sns.jointplot(x= joint_defense, y=joint_atk, alpha=.7)
#what is the distribution of Pokemon HP?
pokemon.HP.plot.hist(bins=20,title='Number of pokemon per HP interval',figsize=(8,4))
#table giving the average Total for the two types
types_table = pokemon.groupby(['Type 1','Type 2']).Total.mean()
types_table.sample(10)
#what type is the strongest on average?
types_table.idxmax()
#what are the average stats for Normal and Legendary Pokemons, respectivley?
norm_vs_leg = pokemon.groupby('Legendary')['HP','Attack','Defense','Sp. Atk','Sp. Def', 'Speed'].mean()
norm_vs_leg.index=['Normal','Legendary']
norm_vs_leg
norm_vs_leg.plot.pie(subplots=True, layout=(2,3), figsize=(16,10),
                     legend=False, colors=['aquamarine','tomato'])
plt.figure(figsize=(12,6))
sns.violinplot(x='Generation', y='Total', hue='Legendary', data=pokemon, split=True)
sns.despine(top=True, right=True)