import pandas as pd

import numpy as np



# Patch statsmodels kdetools

def _revrt(X,m=None):

    """

    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.

    """

    if m is None:

        m = len(X)

    i = int(m // 2+1)

    y = X[:i] + np.r_[0,X[i:],0]*1j

    return np.fft.irfft(y)*m



from statsmodels.nonparametric import kdetools



# replace the implementation with new method.

kdetools.revrt = _revrt

import seaborn as sns

import matplotlib.pyplot as plt
pd.__version__
% matplotlib inline
df = pd.read_csv("../input/Pokemon.csv")

df.columns = ['id','name','type_1','type_2','total',

              'hp','attack','defense','sp_atk',

              'sp_def','speed','generation','legendary']

df.head()
# Statistics for each attribute

df[['total','hp','attack','defense','sp_atk','sp_def','speed']].describe().round(1)
# How common are Pokemon names starting with each letter

sns.countplot(x=df['name'].str[0], order=sorted(df['name'].str[0].unique()))
# How common are Pokemon names ending with each letter

sns.countplot(x=df['name'].str[-1], order=sorted(df['name'].str[-1].unique()))
# How long are the names

name_lengths = df['name'].map(lambda name: len(name))

name_lengths.hist()
# Top 10 Pokemon by total attribute

top_10_total = df.sort_values(ascending=False, by='total').head(10)
(

    top_10_total[['name','hp','attack','defense','sp_atk','sp_def','speed']]

    .set_index('name')

    .plot(kind='barh', stacked=True)

)
# Average total attribute by generation

df.groupby('generation')['total'].mean()
# How many types are there

all_types = list(set(df.type_1.unique()) & set(df.type_2.unique()))

print('Pokemon can have any of the following {} types:'.format(len(all_types)))

print(all_types)
print('Type 1 contains {} unique elements'.format(df['type_1'].nunique()))

print('Type 2 contains {} unique elements'.format(df['type_2'].dropna().nunique()))
type_1_set = set(df['type_1'])

type_2_set = set(df['type_2'].dropna())

# ^ Return a new set with elements in either the set or other but not both.

# Looks like there are no unique types to either column

type_1_set ^ type_2_set
# Variety of Pokemon by type in either type_1 or type_2

types = pd.get_dummies(df['type_1']) + pd.get_dummies(df['type_2'])

df_types = df.merge(types, left_index=True, right_index=True)

df_types[all_types].sum().sort_values(ascending=False)
# Variety by generation

type_count_by_gen = df_types.groupby('generation')[all_types].sum()

type_count_by_gen.style.background_gradient(cmap='cool', axis=1)
# How many Pokemon have either 1 or 2 types

df['type_count'] = 2

df.loc[df['type_2'].isnull(), 'type_count'] = 1

sns.countplot(df['type_count'])
df.groupby(['type_count'])['id'].count()
# How common is each type according to type_1 and type_2 columns

type_fig, (type_ax1, type_ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

sns.countplot(y=df['type_1'], ax=type_ax1)

sns.countplot(y=df['type_2'], ax=type_ax2)
# How common are the combinations

def reduce_type_combinations(pokemon):

    """Remove the permutations of types by sorting type_1 and type_2 for each Pokemon

       before returning a single string value representing the combination.

       For example:

       Pokemon X type_1 = Flying type_2 = Fire

       Pokemon Y type_1 = Fire   type_2 = Flying 

       They will both become 'Fire & Flying'"""

    t1 = pokemon['type_1']

    if pd.isnull(pokemon['type_2']):

        return t1

    t2 = pokemon['type_2']

    sorted_types = sorted([t1, t2])

    return ' & '.join(sorted_types)



df['type_combination'] = df.apply(reduce_type_combinations, axis=1)

more_than_1_type = df[df['type_2'].notnull()]



print('There are {} unique type combinations, including 1 and 2 type Pokemon.'.format(df['type_combination'].nunique()))

print('There are {} unique Pokemon 2 type combinations.'.format(more_than_1_type['type_combination'].nunique()))
# Using Pandas style API for highlighting the min and max for each stat

def highlight_min(s):

    '''

    highlight the minimum in a Series yellow.

    '''

    is_min = s == s.min()

    return ['background-color: red' if v else '' for v in is_min]



def highlight_max(s):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    return ['background-color: green' if v else '' for v in is_max]
# Total statistics for each type, which are NOT part of a combination of types

def stat_summary_for_single_type_pokemon(df, attribute):

    single_type = df[df['type_2'].isnull()]

    single_type = single_type.rename(columns={'type_1': 'type'})



    single_type_stats = (

        single_type

        .groupby('type')[attribute]

        .describe()

        .round(1)

        .reset_index()

        .pivot_table(index='type', columns='level_1', values=attribute)

    )

    

    return single_type_stats
stat_summary_for_single_type_pokemon(df, 'total')
# Total statistics for each type, which ARE part of a combination of types

def stacked_combo_type_attributes(df, attribute):

    if isinstance(attribute, list):

        columns = ['type_combination'] + attribute

        columns1 = ['1'] + attribute

        columns2 = ['2'] + attribute

    else:

        columns = ['type_combination', attribute]

        columns1 = ['1', attribute]

        columns2 = ['2', attribute]

    type_combination_totals = more_than_1_type[columns]

    type_combination_totals['1'], type_combination_totals['2'] = type_combination_totals['type_combination'].str.split(' & ').str



    stacked_type_attributes = (

        type_combination_totals[columns1]

        .rename(columns={'1':'type'})

        .append(

           type_combination_totals[columns2]

           .rename(columns={'2': 'type'})

           )

    )

    

    return stacked_type_attributes



def describe_attribute_combo_type_pokemon(df, attribute):

    stacked_type_attributes = stacked_combo_type_attributes(df, attribute)

    combo_type_stats = (

        stacked_type_attributes

        .groupby('type')[attribute]

        .describe()

        .round(1)

        .reset_index()

        .pivot_table(index='type', columns='level_1', values=attribute)

    )

    

    return combo_type_stats
# Stack single and combos together to see how the rank

# Sorted by mean

def both_single_and_combo_stats(df, attribute):

    single_type_stats = stat_summary_for_single_type_pokemon(df, attribute).reset_index()

    combo_type_stats = describe_attribute_combo_type_pokemon(df, attribute).reset_index()

    combo_type_stats['type'] = combo_type_stats['type'].map(lambda t: "{}+".format(t))



    single_combo_type_stats = (

        combo_type_stats

        .append(single_type_stats)

        .sort_values(ascending=False, by='mean')

        .reset_index()

    )



    return (

        single_combo_type_stats

        .style

        .apply(highlight_min)

        .apply(highlight_max)

    )
# Top 20 common combinations

def top_20_combos_by_attribute(df, attribute, sorted_by='mean'):

    top_20_combos = (

        more_than_1_type

        .groupby('type_combination')[attribute]

        .describe()

        .round(1)

        .reset_index()

        .pivot_table(index='type_combination', columns='level_1', values=attribute)

        .sort_values(ascending=False, by=sorted_by)

        .head(20)

    )



    return (

        top_20_combos

        .style

        .apply(highlight_min)

        .apply(highlight_max)

    )
both_single_and_combo_stats(df, 'total')
top_20_combos_by_attribute(df, 'total', sorted_by='mean')
top_20_combos_by_attribute(df, attribute='total', sorted_by='max')
df.groupby('type_count')['total'].describe().round(2)
df.groupby('type_count')['total'].hist(alpha=0.5)
plt.figure(figsize=(15,10))

sns.boxplot(x='type_1', y='total', hue='type_count', data=df)
# The max hp for a single type pokemon is considerably higher than a combo type

df.groupby('type_count')['hp'].describe().round(2)
df[['name','type_combination','hp']].sort_values(ascending=False, by='hp').head(10)
df.groupby('type_count')['hp'].hist(alpha=0.5)
plt.figure(figsize=(15,10))

sns.boxplot(x='type_1', y='hp', hue='type_count', data=df)
df.groupby('type_count')['attack'].describe().round(2)
df.groupby('type_count')['attack'].hist(alpha=0.5)
plt.figure(figsize=(15,10))

sns.boxplot(x='type_1', y='attack', hue='type_count', data=df)
df.groupby('type_count')['defense'].describe().round(2)
df.groupby('type_count')['defense'].hist(alpha=0.5)
plt.figure(figsize=(15,10))

sns.boxplot(x='type_1', y='defense', hue='type_count', data=df)
df.groupby('type_count')['speed'].describe().round(2)
df.groupby('type_count')['speed'].hist(alpha=0.5)
plt.figure(figsize=(15,10))

sns.boxplot(x='type_1', y='speed', hue='type_count', data=df)
sns.jointplot(df['attack'], df['defense'], kind="hex", size=7, space=0)
stacked_types = stacked_combo_type_attributes(df, ['hp','speed','attack','defense'])
grass = stacked_types[stacked_types['type'] == 'Grass']

water = stacked_types[stacked_types['type'] == 'Water']

fire = stacked_types[stacked_types['type'] == 'Fire']

electric = stacked_types[stacked_types['type'] == 'Electric']
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)



sns.kdeplot(water.attack, water.defense, ax=axes[0, 0],

            cmap="Blues", shade=True, shade_lowest=False)

sns.kdeplot(fire.attack, fire.defense, ax=axes[0, 1],

            cmap="Reds", shade=True, shade_lowest=False)

sns.kdeplot(grass.attack, grass.defense, ax=axes[1, 0],

            cmap="Greens", shade=True, shade_lowest=False)

sns.kdeplot(electric.attack, electric.defense, ax=axes[1, 1],

            cmap="YlOrBr", shade=True, shade_lowest=False)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)



sns.kdeplot(water.speed, water.defense, ax=axes[0, 0],

            cmap="Blues", shade=True, shade_lowest=False)

sns.kdeplot(fire.speed, fire.defense, ax=axes[0, 1],

            cmap="Reds", shade=True, shade_lowest=False)

sns.kdeplot(grass.speed, grass.defense, ax=axes[1, 0],

            cmap="Greens", shade=True, shade_lowest=False)

sns.kdeplot(electric.speed, electric.defense, ax=axes[1, 1],

            cmap="YlOrBr", shade=True, shade_lowest=False)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)



sns.kdeplot(water.hp, water.defense, ax=axes[0, 0],

            cmap="Blues", shade=True, shade_lowest=False)

sns.kdeplot(fire.hp, fire.defense, ax=axes[0, 1],

            cmap="Reds", shade=True, shade_lowest=False)

sns.kdeplot(grass.hp, grass.defense, ax=axes[1, 0],

            cmap="Greens", shade=True, shade_lowest=False)

sns.kdeplot(electric.hp, electric.defense, ax=axes[1, 1],

            cmap="YlOrBr", shade=True, shade_lowest=False)
g = sns.PairGrid(water, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(plt.hist)
g = sns.PairGrid(fire, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Reds_d")

g.map_upper(plt.scatter, color='Red')

g.map_diag(plt.hist)
g = sns.PairGrid(grass, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Greens_d")

g.map_upper(plt.scatter, color='Green')

g.map_diag(plt.hist)
g = sns.PairGrid(stacked_types, hue="type", palette="viridis")

g.map(plt.scatter)

g.add_legend()