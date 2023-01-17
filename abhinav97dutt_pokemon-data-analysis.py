import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
pokemon = pd.read_csv('../input/Pokemon.csv')

pokemon[50:100]
set(pokemon['Type 1'])
pokemon.columns

pokemon.drop(['Type 2'],inplace=True,axis=1)
pokemon
grouped_by_defense = pokemon[pokemon['Defense']> 105].groupby('Type 1')
defense_data = grouped_by_defense.count()['Name'].sort_values(ascending=False)
l = range(len(defense_data))

plt.bar(l,defense_data.data, color='blue', edgecolor='red')

plt.xticks(l,defense_data.index, rotation='90')

plt.show()
grouped_by_attack = pokemon[pokemon['Attack']>100].groupby('Type 1')

attack_data = grouped_by_attack.count()['Name'].sort_values(ascending = False)
l = range(len(attack_data))

plt.bar(l,attack_data.data,color='green')

plt.xticks(l,attack_data.index,rotation='90')

plt.show()
grouped = pokemon.groupby('Type 1')

grouped.count()['Name'].sort_values(ascending=False)
grouped_by_speed = pokemon[pokemon['Speed']> 100].groupby('Type 1') 

speed_data = grouped_by_speed.count()['Name'].sort_values(ascending=False)

speed_data
l = range(len(grouped_by_speed))

plt.bar(l,speed_data.data,color='pink',edgecolor='blue')

plt.xticks(l,speed_data.index,rotation='90')

plt.show()
grouped_by_special = pokemon[pokemon['Sp. Atk']> 110].groupby('Type 1') 

specials_data = grouped_by_special.count()['Name'].sort_values(ascending=False)

specials_data
l = range(len(grouped_by_special))

plt.bar(l,specials_data.data,color='yellow',edgecolor='green')

plt.xticks(l,specials_data.index,rotation='90')

plt.show()


sns.regplot(x=pokemon['Attack'],y=pokemon['Speed'],data=pokemon)

plt.title('Speed Vs. Attack')

plt.show()



sns.regplot(x=pokemon['Speed'],y=pokemon['Defense'],data=pokemon)

plt.title('Defense Vs. Speed')

plt.show()
stats_pokemon = pokemon[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
stats_pokemon.corr()
sns.heatmap(stats_pokemon.corr())