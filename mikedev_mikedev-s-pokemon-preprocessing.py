import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import os
import itertools
print(os.listdir("../input/pokemon-challenge/"))
plt.style.use('ggplot')
combats = pd.read_csv('../input/pokemon-challenge/combats.csv')
display(combats.head())
wins_by_pokemon = combats.groupby('Winner')[['First_pokemon']].count().rename(columns={'First_pokemon': 'num_wins'})
wins_by_pokemon.head()
pokemon = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
pokemon.set_index('#', inplace=True)
display(pokemon.describe())
pokemon['num_wins'] = wins_by_pokemon.num_wins
pokemon['Generation'] = pokemon['Generation'].astype('category')
pokemon.head()
pokemon.head()
numdf = pokemon.select_dtypes(exclude=['object', 'category', 'bool'])
plt.figure(figsize=(15,10))
sns.heatmap(numdf.corr(), cbar=True, annot=True, square=True, fmt='.2f')
plt.show()
y_col = 'num_wins'
X_cols = list(set(numdf.columns) - {y_col})
fig, axes = plt.subplots(ncols=2, nrows=(len(X_cols)//2 + len(X_cols)%2),figsize=(15,15))
for X_col, ax in zip(X_cols, itertools.chain(*axes)):
    sns.regplot(x=X_col, y=y_col, data=pokemon, ax=ax)
plt.show()
pokemon.isnull().sum()
pokemon[pokemon.Name.isnull()]
pokemon.iloc[60:70]
pokemon.loc[63, 'Name'] = 'Primeape'
pokemon.loc[63]
pokemon['num_wins'] = pokemon.num_wins.fillna(0)
def count_wins_by(col):
    grouped = pokemon.groupby(col)[['num_wins']].agg(['sum', 'count'])
    grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'num_wins sum': 'num_wins', 'num_wins count': 'num_pokemon'})
    grouped['normalized_num_wins'] = grouped['num_wins'] / grouped['num_pokemon']
    return grouped
victorytype1 = count_wins_by('Type 1')
ax1 = victorytype1.reset_index().plot(x='Type 1', y='num_wins', kind='bar', title='Victory by pokemon main type', legend=False)
ax1.set_ylabel('#wins')
ax2 = victorytype1.reset_index().plot(x='Type 1', y='normalized_num_wins', kind='bar', title='Victory normalized with num of pokemon\n of the same type by pokemon main type', legend=False)
ax2.set_ylabel('#wins/#pokemon')
plt.show()
pokemon['Type'] = pokemon['Type 1'] + pokemon['Type 2'].fillna('')
pokemon.head()
victorytypes = count_wins_by('Type')
victorytypes = victorytypes.sort_values(by=['num_wins', 'normalized_num_wins'], ascending=False)
ax1 = victorytypes.iloc[:20, :].reset_index().plot(x='Type', y='num_wins', kind='bar', title='Victory by pokemon main type', legend=False)
ax1.set_ylabel('#wins')
ax2 = victorytypes.iloc[:20, :].reset_index().plot(x='Type', y='normalized_num_wins', kind='bar', title='Victory normalized with num of pokemon\n of the same type by pokemon main type', legend=False)
ax2.set_ylabel('#wins/#pokemon')
plt.show()
pokemon = pokemon.join(victorytypes, on='Type', rsuffix='_Type')
pokemon = pokemon.join(victorytype1, on='Type 1', rsuffix='_Type1')
pokemon = pokemon.rename(columns={'normalized_num_wins': 'normalized_num_wins_Type', 'num_pokemon': 'num_pokemon_Type'})
pokemon.head()
pokemon2 = pd.read_csv('../input/pokemon-evolutions/pokemon_species.csv')
pokemon2.set_index('id', inplace=True)
display(pokemon2.describe())
display(pokemon2.head())
evolved = pokemon2.sort_values(by='evolves_from_species_id', ascending=False).dropna(subset=['evolves_from_species_id'])
evolved.evolves_from_species_id.astype(dtype='int', inplace=True)
evolved.head()
def id_to_name(id_pokemon):
    result =  pokemon2.loc[int(id_pokemon), 'identifier']
    assert isinstance(result, str), print(type(result), result)
    return result.capitalize().strip()
if 'num_evolutions' in pokemon.columns:
    del pokemon['num_evolutions']
pokemon_iname = pokemon.set_index('Name')
pokemon_iname['num_evolutions'] = 0
evolved['identifier'] = evolved.identifier.str.capitalize().str.strip()
evolved_iname = evolved.set_index('identifier')
evolved_iname['evolves_from_species_id'] = evolved_iname['evolves_from_species_id'].transform(id_to_name)
for evolved_pokemon in evolved_iname.itertuples():
    if evolved_pokemon.evolves_from_species_id in pokemon_iname.index and evolved_pokemon.Index in pokemon_iname.index:
        pokemon_iname.loc[evolved_pokemon.evolves_from_species_id, 'num_evolutions']+= pokemon_iname.loc[evolved_pokemon.Index, 'num_evolutions'] + 1
pokemon = pokemon.join(pokemon_iname[['num_evolutions']], on='Name')
del pokemon_iname, evolved_iname
pokemon.head()
is_mega = pokemon.Name.str.startswith('Mega ')
has_megaevol_names = pokemon.loc[is_mega, 'Name'].transform(lambda name: name.split()[1])
print(has_megaevol_names[:5])
pokemon['is_mega'] = is_mega
pokemon['has_mega_evolution'] = pokemon['Name'].isin(has_megaevol_names)
pokemon.head(10).T
pokemon[['num_evolutions']].describe()
pokemon.query('num_evolutions==8')
pokemon.boxplot(by='num_evolutions', column='num_wins', figsize=(10, 10))
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14,9))
pokemon.boxplot(by='Legendary', column='num_wins', figsize=(5,5), ax=axs[0])
pokemon.boxplot(by='is_mega', column='num_wins', figsize=(5,5), ax=axs[1])
pokemon.boxplot(by='has_mega_evolution', column='num_wins', figsize=(5,5), ax=axs[2])
pokemon['has_second_type'] = ~pokemon['Type 2'].isna()
pokemon.head()
print(pokemon.columns)
display(pokemon.head())
pokemon.drop(columns=['Name'], inplace=True)
catcols = ['Type 1', 'Type 2', 'Type' ]
pokemon[catcols] = pokemon[catcols].transform(lambda catcol: catcol.astype('category').cat.codes, axis=1)
pokemon.head().T
numvars = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'num_wins', 'num_wins_Type', 'num_pokemon_Type', 'normalized_num_wins_Type', 'num_wins_Type1', 'num_pokemon_Type1', 'normalized_num_wins_Type1']
pokemon[numvars] = pokemon[numvars].transform(lambda col: (col - col.mean()) / (col.std()))
plt.figure(figsize=(15,10))
sns.heatmap(pokemon[numvars].corr(), cbar=True, annot=True, square=True, fmt='.2f')
pokemon.drop(columns=['num_wins_Type', 'num_wins_Type1'], inplace=True)
bool_cols = pokemon.select_dtypes('bool').columns
pokemon[bool_cols] = pokemon[bool_cols].transform(lambda col: col.astype('int'))
pokemon.head()
pokemon.to_csv('pokemon_mikedev_preprocessed.csv')
combats.to_csv('combats.csv')
!cp ../input/pokemon-challenge/tests.csv ./
# Check that every id of Winner is on the column First_pokemon or in Second_pokemon
assert ((combats.Winner == combats.First_pokemon) | (combats.Winner == combats.Second_pokemon)).all()
combats['Winner'] = (combats.Winner == combats.Second_pokemon).astype('int')
combats.head()
pokemon.head()
test = pd.read_csv('../input/pokemon-challenge/tests.csv')
display(test.head())
train_test = pd.concat([combats, test], join='inner')
train_len = len(combats)
train_test = \
    train_test\
        .merge(pokemon, left_on='First_pokemon', right_index=True, how='left')\
        .merge(pokemon, left_on='Second_pokemon', right_index=True, suffixes=('_first_pokemon', '_second_pokemon'), how='left')
statcols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
for statcol in statcols:
    train_test[statcol + '_diff'] = train_test[statcol+'_first_pokemon'] - train_test[statcol+'_second_pokemon']
y = combats.Winner.values
combats = train_test.iloc[:train_len, :]
test = train_test.iloc[train_len:, :]
combats['Winner'] = y
display(combats.head())
display(test.head())
del train_test
combats.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)