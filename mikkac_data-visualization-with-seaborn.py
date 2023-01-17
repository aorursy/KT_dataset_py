# Notebook created while doing the tutorial: https://elitedatascience.com/python-seaborn-tutorial



import numpy as np

import pandas as pd

import seaborn as sns

from  matplotlib import pyplot as plt

%matplotlib inline
df = pd.read_csv("https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv", index_col=0)



df.head()
sns.lmplot(data=df, x='Attack', y='Defense', size=5, aspect=2)
sns.lmplot(data=df, x='Attack', y='Defense', fit_reg=False, hue='Legendary', size=5, aspect=2)

plt.xlim(0, None)

plt.ylim(0, None)
df_box = df.drop(['Total', 'Generation', 'Legendary'], axis=1)

sns.boxplot(data=df_box)
sns.set_style('whitegrid')

sns.violinplot(data=df, x='Type 1', y='Attack')
pkmn_type_colors = ['#78C850',  # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                   ]



sns.violinplot(data=df, x='Type 1', y='Attack', palette=pkmn_type_colors)
sns.swarmplot(data=df, x='Type 1', y='Attack', palette=pkmn_type_colors)
plt.figure(figsize=(16,10))

sns.violinplot(data=df, x='Type 1', y='Attack', palette=pkmn_type_colors, inner=None)

sns.swarmplot(data=df, x='Type 1', y='Attack', color='k', alpha=0.6)
df_box.head()

melted_df = pd.melt(df_box, id_vars=['Name', 'Type 1', 'Type 2'], var_name='Stat')

melted_df.head()
plt.figure(figsize=(20,10))

sns.swarmplot(data=melted_df, x='Stat', y='value', hue='Type 1')
plt.figure(figsize=(20,10))

sns.swarmplot(data=melted_df, x='Stat', y='value', hue='Type 1', split=True, palette=pkmn_type_colors)

plt.ylim(0, None)

plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.figure(figsize=(10,5))

corr = df_box.corr()

sns.heatmap(corr)
plt.figure(figsize=(10,5))

sns.distplot(df.HP) # histogram
# bar plot

plt.figure(figsize=(10,5))

sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)

plt.xticks(rotation=-45)
plt.figure(figsize=(10,5))

g = sns.factorplot(x='Type 1', y='Attack', data=df, hue='Legendary', col='Legendary', kind='swarm')

g.set_xticklabels(rotation=-45)
# density plot

plt.figure(figsize=(10,5))

sns.kdeplot(df.Defense, df.Speed)

plt.xlim(0, 150)

plt.ylim(0, 150)
# joint distribution plot

plt.figure(figsize=(10,5))

sns.jointplot(x='Defense', y='Speed', data=df)
plt.figure(figsize=(10,5))

sns.jointplot(data=df, x='Attack', y='Defense', kind='kde')
import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline

import numpy as pd

import pandas as pd
rng = np.random.RandomState(0)

x = np.linspace(0, 10, 500)

y = np.cumsum(rng.randn(500, 6), axis=0)

print(x.shape, y.shape)
plt.plot(x, y)

plt.legend('ABCDEF', ncol=2, loc='upper_left')
import seaborn as sns

sns.set()

plt.plot(x, y)

plt.legend('ABCDEF', ncol=2, loc='upper_left')
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size = 2000)

data= pd.DataFrame(data, columns=['x', 'y'])



for col in data.columns:

    plt.hist(data[col], normed=True, alpha=0.5)

for col in data.columns:

    sns.kdeplot(data[col], shade=True)
for col in data.columns:

    sns.distplot(data[col])
sns.kdeplot(data)
with sns.axes_style('white'):

    sns.jointplot(x='x', y='y', data=data, kind='kde')
with sns.axes_style('white'):

    sns.jointplot(x='x', y='y', data=data, kind='hex')
iris = sns.load_dataset('iris')

iris.head()
sns.pairplot(iris, hue='species', size=2.5)
tips = sns.load_dataset('tips')

tips.head()
tips['tip_pct'] = 100 * tips.tip / tips.total_bill

grid = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)

grid.map(plt.hist, 'tip_pct', bins=np.linspace(0, 40, 15))
with sns.axes_style(style='ticks'):

    g = sns.factorplot(x='day', y='total_bill', hue='sex', data=tips, kind='box')

    g.set_axis_labels('Day', 'Total Bill')
with sns.axes_style(style='white'):

    sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
with sns.axes_style(style='white'):

    sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
planets = sns.load_dataset('planets')

planets.head()
with sns.axes_style(style='white'):

    g = sns.factorplot(x='year', data=planets, aspect=2, kind='count', color='steelblue')

    g.set_xticklabels(step=5)
with sns.axes_style(style='white'):

    g = sns.factorplot(x='year', data=planets, size=5, aspect=3.0, kind='count', hue='method', order=range(2001, 2015))

    g.set_ylabels('Number of planets discovered')
data = pd.read_csv('https://raw.githubusercontent.com/jakevdp/marathon-data/master/marathon-data.csv',

                  converters={'split':pd.to_timedelta, 'final':pd.to_timedelta})

data.head()
data['split_sec'] = data.split.astype(int) / 1E9

data['final_sec'] = data.final.astype(int) / 1E9

data.head()
plt.figure(figsize=(20,10))

with sns.axes_style(style='white'):

    g = sns.jointplot(x='split_sec', y='final_sec', data=data, kind='hex')

    g.ax_joint.plot(np.linspace(4000, 16000), np.linspace(8000, 32000), ':k')
data['split_frac'] = 1 - 2 * data.split_sec / data.final_sec

data.head()
sns.distplot(data.split_frac, kde=False)

plt.axvline(0, color='k', linestyle='--')
sum(data.split_frac < 0) # 'negative-split' - pace increasing throughout the race, instead of decreasing
g = sns.PairGrid(data=data, vars=['age', 'split_sec', 'final_sec', 'split_frac'], hue='gender', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend()
sns.kdeplot(data.split_frac[data.gender=='W'], label='women', shade=True)

sns.kdeplot(data.split_frac[data.gender=='M'], label='men', shade=True)
sns.violinplot(x='gender', y='split_frac', data=data, palette=['lightblue', 'lightpink'])
data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))

data.head()
plt.figure(figsize=(10,5))

with sns.axes_style(style=None):

    sns.violinplot(x='age_dec', y='split_frac', hue='gender', data=data,

                   split=True, inner='quartile', palette=['lightblue', 'lightpink'])
g = sns.lmplot(x='final_sec', y='split_frac', col='gender', data=data,

              markers='.', scatter_kws=dict(color='c'))

g.map(plt.axhline, y=0.1, color='k', ls=':')