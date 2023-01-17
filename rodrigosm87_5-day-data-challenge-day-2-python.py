import pandas as pd

import matplotlib.pyplot as plt



dt = pd.read_csv('../input/Pokemon.csv')
plt.title('Pokemons histogram by attack')

plt.hist(dt['Attack'])
plt.title('Pokemons histogram by defense')

plt.hist(dt['Defense'])
plt.title('Pokemons histogram by generation')

plt.hist(dt['Generation'])