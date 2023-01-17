import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import ast



plt.style.use('seaborn-ticks')
pokemon_data = pd.read_csv('../input/pokemon-data.csv', 

                           sep = ';', converters={'Types':ast.literal_eval, 'Abilities':ast.literal_eval, 'Moves':ast.literal_eval})

move_data = pd.read_csv('../input/move-data.csv', index_col = 0)

for var in ['Power', 'Accuracy']:

    move_data[var].replace('None', np.nan, inplace=True)

    move_data[var] = move_data[var].astype(float)
pokemon_data.head()
plt.scatter(pokemon_data.HP, pokemon_data.Attack, alpha = 0.3)

plt.xlabel('HP')

plt.ylabel('Attack')

plt.show()
moves_list = []

for moves in pokemon_data.Moves:

    moves_list.append(len(moves))

pokemon_data['Number of Moves'] = moves_list
pokemon_data['Number of Moves'] = [len(a) for a in pokemon_data.Moves]
plt.scatter(pokemon_data['Attack'], pokemon_data['Special Attack'], 

            c = pokemon_data['Number of Moves'], alpha = 0.3, cmap = 'viridis')

c = plt.colorbar()

c.set_label('Number of Moves')

plt.xlabel('Attack')

plt.ylabel('Special Attack')

plt.show()
move_data.head()
plt.hist(move_data.Power, range = (0, move_data.Power.max()), bins = 20)

plt.xlabel('Power')

plt.ylabel('Number of Moves')

plt.show()
sns.barplot('Type', 'Power', data = move_data)

plt.xticks(rotation = 45)

plt.show()
sns.swarmplot('Type', 'Power', data = move_data, edgecolor = 'black', linewidth = 1)

plt.xticks(rotation = 45)

plt.show()
sns.barplot('Type', 'Power', data = move_data)

sns.swarmplot('Type', 'Power', data = move_data, edgecolor = 'black', linewidth = 1, alpha = 0.5)

plt.xticks(rotation = 45)

plt.show()
for contest, marker in zip(set(move_data.Contest), ['o', 'x', 'D', '+', 'v', '*']):

    plt.scatter('Power', 'PP', data = move_data[move_data.Contest == contest], marker = marker, alpha = 0.5, label = contest)

plt.xlabel('Power')

plt.ylabel('PP')

plt.legend()

plt.show()