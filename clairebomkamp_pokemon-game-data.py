import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import os



plt.style.use('seaborn-ticks')



# This is some sample data, but you can upload your own instead if you like!

game_data = pd.read_csv("../input/1552513645.59.csv")

game_data.head()
pokemon = 'Jigglypuff'

game_data.Event == pokemon
game_data[game_data.Event == pokemon]
for pokemon in set(game_data.Event):

    game_data[pokemon] = (game_data.Event == pokemon).cumsum()
game_data.head()
for pokemon in set(game_data.Event):

    plt.plot(game_data.Time, game_data[pokemon], label = pokemon)

plt.xlabel('Time')

plt.ylabel('Number caught')

plt.legend()

plt.show()
file_path = "../input/"

files = []

for f in os.listdir(file_path):

    df = pd.read_csv(file_path + f, index_col = 0)

    if df.shape[1] == 5:

        df.drop('Unnamed: 0.1', axis = 1, inplace = True)

        df.to_csv(file_path + f)

    files.append(df)



game_data = pd.concat(files, sort = False)
pokemon = 'Jigglypuff'

subset = game_data[game_data.Event == pokemon].copy()
subset.head()
screen = np.zeros((19, 19))



for x in range(19):

    for y in range(19):

        screen[y, x] = subset[(subset.X == x + 1) & (subset.Y == y + 1)].shape[0]
plt.imshow(screen, cmap = 'viridis', origin = 'lower')

c = plt.colorbar()

c.set_label('Number of ' + pokemon + 's seen')

plt.axis('off')

plt.show()
sns.violinplot(x = 'Event', y = 'X', data = game_data)

plt.gca().set_xlabel('')

plt.show()
subset = game_data[game_data.Event == 'Spearow'].copy()

subset2 = game_data[game_data.Event == 'Nidorina'].copy()



sns.kdeplot(subset2.X, subset2.Y, cmap = 'Reds_d')

sns.kdeplot(subset.X, subset.Y, cmap = 'Blues_d')

plt.show()