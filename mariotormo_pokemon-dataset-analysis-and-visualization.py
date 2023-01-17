# Data analysis and wrangling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
%matplotlib inline
pokedex = pd.read_csv('../input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv')
pokedex = pokedex.drop(['Unnamed: 0'], axis=1)
pokedex.info()
pokedex.head(10)
null_filter = pokedex.isna().sum()
missing_values = null_filter.loc[pokedex.isna().sum() != 0].to_frame().copy()
missing_values
missing_values.columns = ['count']
missing_values['Name'] = missing_values.index
missing_values.reset_index(drop=True, inplace=True)
missing_values

sns.barplot(x='Name', y='count', data=missing_values);
plt.xticks(rotation=90);

ax = sns.catplot(x="generation", kind="count", data=pokedex);
ax.set(xlabel='Generation', ylabel='Nr of Pokemon', title='Number of Pokemon in each Generation');
sns.catplot(x="generation", col="status", kind="count", data=pokedex);
poke_filter = pokedex.loc[pokedex.status != "Normal"]
sns.catplot(x="generation", col="status", kind="count", data=poke_filter);
pokedex.species.value_counts()
pokedex.groupby('generation').species.describe()
sns.catplot(x='generation', y='height_m', data=pokedex);
pok_height_out = pokedex[pokedex.height_m < 20]
sns.catplot(x='generation', y='height_m', data=pok_height_out);
sns.catplot(x='generation', y='height_m', kind='box', data=pok_height_out);
pokedex.height_m.mean()
sns.catplot(x='generation', y='weight_kg', data=pokedex);

sns.catplot(x='generation', y='weight_kg', kind='box', data=pokedex);
ax = sns.relplot(x='height_m', y='weight_kg', hue='generation', legend='full',palette='Set1', data=pokedex);
ax.set(xlim=(0, None), ylim=(0, None));
sns.set_style('whitegrid')
g = sns.relplot(x='pokedex_number', y='attack', kind="line", hue = 'generation', palette='Set1', height = 8, aspect = 4, data=pokedex)
g.set(xlim=(0, None));
# g.fig.autofmt_xdate()
g = sns.relplot(x='pokedex_number', y='defense', kind="line", hue = 'generation', palette='Set1', height = 8, aspect = 4, data=pokedex)
g.set(xlim=(0, None));
g.fig.autofmt_xdate()
ax = sns.relplot(x='attack', y='defense', hue='generation', legend='full',palette='Set1', data=pokedex);
ax.set(xlim=(0, None), ylim=(0, None));