import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.simplefilter("ignore")
df = pd.read_csv('../input/pokemon/Pokemon.csv', encoding = 'unicode_escape', index_col = 0)
sns.lmplot(x = "Attack", y = "Defense", data = df);
sns.lmplot(x = "Attack", y = "Defense", fit_reg = False, data = df);
sns.lmplot(x = "Attack", y = "Defense", hue = "Stage",fit_reg = False, data = df);
plt.figure(figsize = (8,5));

sns.boxplot(data = df);
df_stats = df.drop(["Total", "Stage", "Legendary"], axis = 1)

df_stats
sns.boxplot(data = df_stats);
plt.figure(figsize = (12,6));

sns.set_style("whitegrid")

sns.violinplot(x = "Type 1", y = "Attack", data = df);
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
plt.figure(figsize = (12,6));

sns.violinplot(x = "Type 1", y = "Attack", data = df, palette = pkmn_type_colors);
plt.figure(figsize = (12,6));

sns.swarmplot(x = "Type 1", y = "Attack", data = df, palette = pkmn_type_colors);
plt.figure(figsize = (12,6));

sns.violinplot(x = "Type 1", y = "Attack", data = df, palette = pkmn_type_colors, inner = None);

sns.swarmplot(x = "Type 1", y = "Attack", data = df, color = "k", alpha = 0.7);

plt.title('Attack by Type');
df_stats.head()
df_melted = pd.melt(df_stats, 

                    id_vars = ["Name", "Type 1", "Type 2"], 

                    var_name = "Stat")

df_melted.head()
plt.figure(figsize=(10,6));

sns.swarmplot(x = "Stat", y = "value", hue = "Type 1", data = df_melted, split = True, palette = pkmn_type_colors);

plt.ylim(0, 260);

plt.legend(bbox_to_anchor = (1,1), loc = 2);
corr = df_stats.corr()

sns.heatmap(corr);
sns.distplot(df.Attack);
sns.countplot(x = "Type 1", data = df, palette = pkmn_type_colors);

plt.xticks(rotation = -45);
df_stages = sns.factorplot(x = "Type 1", y = "Attack", hue = "Stage", col = "Stage", kind = "swarm", data = df);

df_stages.set_xticklabels(rotation = -45);
sns.kdeplot(df.Attack, df.Defense);
sns.jointplot(x='Attack', y='Defense', data=df);