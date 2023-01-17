import numpy   as np
import pandas  as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

pokemon = pd.read_csv('../input/pokemon.csv')
combats = pd.read_csv('../input/combats.csv')

pokemon.rename(columns={'#': 'Id'}, inplace=True)
combats.rename(columns={'First_pokemon': '1st', 'Second_pokemon': '2nd'}, inplace=True)

# Setting the Id column to be the index makes joins convenient
pokemon.set_index('Id', inplace=True)
generic_palette = sns.color_palette();
pokemon_types_palette = {
    'Normal':   '#A8A77A',
    'Fire':     '#EE8130',
    'Water':    '#6390F0',
    'Electric': '#F7D02C',
    'Grass':    '#7AC74C',
    'Ice':      '#96D9D6',
    'Fighting': '#C22E28',
    'Poison':   '#A33EA1',
    'Ground':   '#E2BF65',
    'Flying':   '#A98FF3',
    'Psychic':  '#F95587',
    'Bug':      '#A6B91A',
    'Rock':     '#B6A136',
    'Ghost':    '#735797',
    'Dragon':   '#6F35FC',
    'Dark':     '#705746',
    'Steel':    '#B7B7CE',
    'Fairy':    '#D685AD',
}

height = 7
width  = 20
figsize = (width, height)
aspect_ratio = width / height

matplotlib.rcParams['figure.figsize'] = figsize
sns.set_style('ticks')
# I'm gonna add a Loser column just to make things easier later
combats['Loser'] = np.where(combats['1st'] != combats['Winner'], combats['1st'], combats['2nd'])
combats.head()
def count_combats_by(column_in_combats, column_in_counts):
    counts = count_by(combats, column_in_combats, column_in_counts)
    return pad_missing_pokemons_with_zeroes(counts, column_in_counts)

def count_by(df, column_in_df, column_in_counts):
    counts = df.groupby(column_in_df).size().to_frame(column_in_counts)
    counts.index.rename('PokemonId', inplace=True)
    return counts
    
def pad_missing_pokemons_with_zeroes(subset, column_name):
    left_out_pokemons = pd.DataFrame({
        'PokemonId':    pokemon.index.difference(subset.index),
        column_name:    0
    })
    left_out_pokemons.set_index('PokemonId', inplace=True)
    return subset.append(left_out_pokemons)

number_of_wins = count_combats_by('Winner', 'NumberOfWins')
number_of_wins.head()
count_by_1st = count_combats_by('1st', 'NumberOf1st')
count_by_2nd = count_combats_by('2nd', 'NumberOf2nd')

number_of_fights = count_by_1st.join(count_by_2nd)
number_of_fights['NumberOfFights'] = number_of_fights['NumberOf1st'] + number_of_fights['NumberOf2nd']
number_of_fights.head()
record = number_of_wins.join(number_of_fights)
record['1stRate'] = record['NumberOf1st']  / record['NumberOfFights'] # Also calculate these
record['2ndRate'] = record['NumberOf2nd']  / record['NumberOfFights'] # as we'll need them later
record['WinRate'] = record['NumberOfWins'] / record['NumberOfFights']
record.head()
merged = pokemon.join(record)
merged.head()
merged.nlargest(5, 'WinRate')
merged.nsmallest(5, 'WinRate')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

sns.boxplot(
    x='Generation',
    y='WinRate', 
    data=merged,
    palette=generic_palette,
    ax=ax1
);

sns.boxplot(
    x='Legendary',
    y='WinRate', 
    data=merged,
    palette=generic_palette,
    ax=ax2
);
merged[(merged['Legendary'] == True) & (merged['WinRate'] < 0.4)]
type1_winrate = merged.loc[: , ('Type 1', 'WinRate')]
type2_winrate = merged.loc[: , ('Type 2', 'WinRate')]

type1_winrate.rename(columns={'Type 1': 'Type'}, inplace=True)
type2_winrate.rename(columns={'Type 2': 'Type'}, inplace=True)

type_winrate = pd.concat([type1_winrate, type2_winrate], ignore_index=True)
type_winrate.dropna(axis=0, how='any', inplace=True)
type_winrate.head()
sns.boxplot(
    x='Type',
    y='WinRate', 
    data=type_winrate,
    palette=pokemon_types_palette,
    order=sorted(type_winrate.Type.unique()),
);
is_rock = (merged['Type 1'] == 'Rock') | (merged['Type 2'] == 'Rock')
is_good = (merged['WinRate'] > 0.8)
merged[is_rock & is_good]
stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']

def graph_pair_stats(x_vars):
    g = sns.PairGrid(
        merged,
        x_vars=x_vars,
        y_vars=['WinRate'],
        hue='Legendary',
        palette=generic_palette,
        size=height
    )
    g = g.map(plt.scatter)
    g.add_legend()
    g.set(xlim=(0, None), ylim=(0, 1))

graph_pair_stats(['Attack',  'Defense'])
graph_pair_stats(['Sp. Atk', 'Sp. Def'])
graph_pair_stats(['HP',      'Speed'])
combat_stats  = combats \
        .merge(pokemon, left_on='Winner', right_index=True) \
        .merge(pokemon, left_on='Loser',  right_index=True, suffixes=[' Of Winner', ' Of Loser'])
combat_stats.head()
combat_speeds = combat_stats[['Speed Of Winner', 'Speed Of Loser']]
combat_speeds.head()
speedy_won = (combat_speeds['Speed Of Winner'] > combat_speeds['Speed Of Loser']).value_counts()
speedy_won
int(speedy_won[True]) / len(combats)