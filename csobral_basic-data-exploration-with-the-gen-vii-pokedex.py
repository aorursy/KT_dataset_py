import pandas as pd

df = pd.read_json('/kaggle/input/pokemon-gen-vii-pokedex/pokedex.json')
n_rows, n_columns = df.shape

print(f'The dataset contains {n_rows} rows and {n_columns} columns')
df.head()
df.dtypes
n_dualtype = len(df[df['Type 2'] != 'None'].index)

frac_dualtype = n_dualtype/df.shape[0]

print(f'There are {n_dualtype} dual-type Pokémon out of {df.shape[0]},'

    f' i.e. {100*frac_dualtype:.2f}% of all Pokémon are dual-typed')
df['Type 1'].nunique()
df['Type 1'].unique()
type_list = list(df.groupby(['Type 1', 'Type 2']).groups.keys())

for tpl in type_list:

    rev_tpl = tpl[::-1]

    if rev_tpl in type_list:

        type_list.remove(rev_tpl)

print(f'The number of unique type combinations used is {len(type_list)}')
print(type_list)
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')
df['Type 2 Replaced'] = df['Type 2'].replace('None', df['Type 1'])

type_chart = df.groupby(['Type 1', 'Type 2 Replaced']).size().unstack()

type_chart.fillna(0, inplace=True)

type_chart.rename_axis(columns={'Type 2 Replaced':'Type 2'}, inplace=True)



unique_type_values = np.tril(type_chart.values)

unique_type_values += np.triu(type_chart.values, k=1).transpose()

type_chart[:] = unique_type_values



mask = np.zeros_like(type_chart, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

mask[np.diag_indices_from(mask)] = False



_, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(type_chart, annot=True, ax=ax, linewidths=1, cmap="PuBu", mask=mask, square=True, cbar=False)