import pandas as pd

import matplotlib.pyplot as plt



plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10, 7)

plt.rcParams['image.cmap'] = 'viridis'



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
pokemons = pd.read_csv('../input/pokemon.csv')

moves = pd.read_csv('../input/moves.csv')

movesets = pd.read_csv('../input/movesets.csv')
corr = pokemons.corr()

plt.matshow(corr)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns, rotation=0)

plt.grid(False)
pokemon_set = pokemons.sample(5, random_state=0)

LVL = 30

pokemon_set['level'] = LVL

pokemon_set = pokemon_set.merge(movesets, on=['ndex', 'species', 'forme'])

def filter_move(move):

    if pd.isnull(move):

        return False

    condition = move.split('-')[0].strip()

    if condition == 'Start':

        return True

    if condition[0] == 'L':

        return int(condition[1:]) >= LVL

    # Ignoring Egg, TM and ORAS moves for now 

    return False

has_move = lambda lm : [m.split('-')[1] for m in filter(filter_move, lm)]

move_columns = [c for c in pokemon_set.columns if 'move' in c]

pokemon_set['moves'] = pokemon_set[move_columns].apply(has_move, axis=1)

important_columns = ["moves", "level"] + ["id", "species", "forme", "ndex", "type1", "type2", "hp", "attack", "defense", "spattack", "spdefense", "speed", "total"]

pokemon_set = pokemon_set[important_columns]

pokemon_set