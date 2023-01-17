import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/pokemon/PokemonData.csv')
df.head()
plt.figure(figsize=(16,12))
sns.lmplot(x='Attack',y='Defense',data=df) ## seaborn and matplotlib
sns.lmplot(x='Attack',y='Defense',data=df,fit_reg=False,hue='Generation')
plt.figure(figsize=(16,12))
sns.boxplot(data=df)
plt.figure(figsize=(16,12))
stats_df=df.drop(['Generation','Legendary'],axis=1)
sns.boxplot(data=stats_df)
plt.figure(figsize=(16,12))
sns.set_style('whitegrid')
sns.violinplot(x='Type1',y='Attack',data=df)
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
plt.figure(figsize=(16,12))
sns.violinplot(x='Type1',y='Attack',data=df,palette=pkmn_type_colors)
df.head(50)
plt.figure(figsize=(16,16))
sns.swarmplot(x='Type1',y='Attack',data=df,palette=pkmn_type_colors)
# Set figure size with matplotlib
plt.figure(figsize=(16,16))
 
# Create plot
sns.violinplot(x='Type1',
               y='Attack', 
               data=df, 
               inner=None, # Remove the bars inside the violins
               palette=pkmn_type_colors)
 
sns.swarmplot(x='Type1', 
              y='Attack', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
 
# Set title with matplotlib
plt.title('Attack by Type')
df.head(6)
# Melt DataFrame
melted_df = pd.melt(stats_df, 
                    id_vars=["Name", "Type1", "Type2","HP"], # Variables to keep
                    var_name="Stat") # Name of melted variable

melted_df.head(30)
print(df.shape) #Original Data Frame
print(melted_df.shape) #Melted Data Frame
plt.figure(figsize=(16,16))
sns.swarmplot(x='Stat',y='value',data=melted_df,hue='Type1')
df.head(10)
plt.figure(figsize=(16,16)) # 1. Sizing the plot

sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type1', 
              dodge=True, # 2. Separate points by hue
              palette=pkmn_type_colors) # 3. Use Pokemon palette

plt.legend(bbox_to_anchor=(1, 1), loc=2) # 4. setting the box to the right
plt.figure(figsize=(16,16))

# calculate correlation

corr = df.corr()

# Plotting heatmap using Pokedex dataset!!!
sns.heatmap(corr)
# Distribution Plot

plt.figure(figsize=(16,12)) # Figure Size

sns.distplot(df.Attack)
# Bar plots help you visualize the distributions of categorical variables.

# count plot

plt.figure(figsize=(16,9))

sns.countplot(x='Type1',data=df, palette=pkmn_type_colors)

# changing rotations of labels

plt.xticks(rotation=-45)
df.head(7)
plt.figure(figsize=(16,14))
sns.kdeplot(df.Attack, df.Defense)
plt.figure(figsize=(16,14))
sns.jointplot(x='Attack', y='Defense', data=df)