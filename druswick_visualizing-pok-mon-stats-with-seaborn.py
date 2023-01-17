import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
pkmn = pd.read_csv('../input/Pokemon.csv')

pkmn_filtered = pkmn[pkmn['Type 1'].isin(['Fire', 'Fairy', 'Dragon', 'Normal','Electric'])]
sns.boxplot(x='Type 1', y='Attack', data=pkmn_filtered)
sns.jointplot(x="Attack", y='Sp. Atk', data=pkmn);
pkmn['Type 2'].val
def has_second(row):

    return not row['Type 2'] == ""

pkmn['Has Second Type'] = pkmn.apply(has_second, axis=1)

sns.boxplot(x='Has Second Type', y='Attack', data=pkmn)