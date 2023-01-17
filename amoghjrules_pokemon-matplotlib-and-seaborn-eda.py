import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/pokemon/Pokemon.csv",index_col=0,encoding='latin1')

df.head()
#comparing attack stats vs. defense stats

sns.lmplot(x='Attack', y='Defense', data=df)
#hue allows addition of another dimension, fit_reg=False removes regression line

sns.lmplot('Attack','Defense',data = df,hue='Stage',fit_reg=False)

df_dropped = df.drop(['Total','Stage','Legendary'],axis=1)

sns.boxplot(data=df_dropped)

sns.set_style('whitegrid')
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
sns.violinplot(x = 'Type 1', y ='Attack',data=df,palette =pkmn_type_colors )
#swarm plots 

sns.swarmplot(x='Type 1', y ='Attack',data =df, palette = pkmn_type_colors)
#swarm + violin plots

## figure size

plt.figure(figsize=(10,6))



sns.violinplot(x = 'Type 1', y ='Attack',inner=None,data=df,palette =pkmn_type_colors )

sns.swarmplot(x='Type 1', y ='Attack', data=df,color='black', alpha=0.7)
#melted dataframe

melted_df = pd.melt(df_dropped, 

                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep

                    var_name="Stat")

melted_df.head()
# 1. Enlarge the plot

plt.figure(figsize=(10,6))

 

sns.swarmplot(x='Stat', 

              y='value', 

              data=melted_df, 

              hue='Type 1', 

              split=True, # 2. Separate points by hue

              palette=pkmn_type_colors) # 3. Use Pokemon palette

 

# 4. Adjust the y-axis

plt.ylim(0, 260)

 

# 5. Place legend to the right

plt.legend(bbox_to_anchor=(1, 1), loc=2)
#heatmap

corr = df_dropped.corr()

sns.heatmap(corr)
sns.distplot(df.Attack)

sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)

plt.xticks(rotation=-45)

# Factor Plot

g = sns.factorplot(x='Type 1', 

                   y='Attack', 

                   data=df, 

                   hue='Stage',  # Color by stage

                   col='Stage',  # Separate by stage

                   kind='swarm') # Swarmplot

 

# Rotate x-axis labels

g.set_xticklabels(rotation=-45)
# Density Plot

sns.kdeplot(df.Attack, df.Defense)
sns.jointplot(x='Attack', y='Defense', data=df)
