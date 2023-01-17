# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pokemon_clean = pd.read_csv('../input/pokemon.csv')

pokemon_clean.type1
pokemon = pokemon_clean.copy()

stats = ['hp', 'attack', 'defense', 'spattack', 'spdefense', 'speed']

pokemon = pokemon[np.logical_and(722 <= pokemon.id, pokemon.id <= 784)]

pokemon['sweep'] = pokemon.speed + pokemon.attack + pokemon.spattack

pokemon = pokemon[np.logical_or(pokemon.type1 == 'Fighting', pokemon.type2 == 'Fighting')]

pokemon['total'] = pokemon[stats].sum(axis=1)

pokemon.sort_values(by='sweep', ascending=False, inplace=True)

pokemon[['species'] + stats + ['total']]