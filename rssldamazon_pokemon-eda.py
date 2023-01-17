import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pokemon_data = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon_data.head()
pokemon_data.info() 
pokemon_data.describe()
# There are no duplicate pokemons
pokemon_data['Name'].value_counts().sort_values(ascending = False)
plt.hist(pokemon_data['Speed'])
plt.hist(pokemon_data['Attack'])
plt.hist(pokemon_data['Defense'])
plt.hist(pokemon_data['Sp. Atk'])
plt.hist(pokemon_data['Sp. Def'])
# Let's plot a scatter matrix but before that we need to convert boolean values to categoricals/numeric values
pokemon_data['Legendary'] = pokemon_data['Legendary'].map({True:1, False:0})

pd.plotting.scatter_matrix(pokemon_data, hist_kwds={'bins':20})
pokemon_data.corr()
# Let's plot a correlation matrix
sns.heatmap(pokemon_data.corr(), annot = True)
# We have 6 types for generations
pokemon_data['Generation'].unique()
pokemon_data.boxplot(column = ['Total'], by = 'Generation')
plt.title('')
plt.suptitle('')
plt.xlabel('Generation')
plt.ylabel('Pokemon strength')
# Let's first check the unique types for type 1 of a pokemon
pokemon_data['Type 1'].unique()
# Defining a boxplot function to follow DRY principle
def generate_boxplot(dataframe, yval, xval):
    dataframe.boxplot([yval], by = xval)
    plt.suptitle('')
    plt.figsize = (40, 10)
    plt.xticks(fontsize = 9, fontname = 'Comic Sans MS')
    plt.tick_params(axis ='x', rotation = 45) 
generate_boxplot(pokemon_data, 'Total', 'Type 1')
generate_boxplot(pokemon_data, 'Total', 'Type 2')
generate_boxplot(pokemon_data, 'Attack', 'Type 1')
generate_boxplot(pokemon_data, 'Defense', 'Type 1')
generate_boxplot(pokemon_data, 'Speed', 'Type 1')
generate_boxplot(pokemon_data, 'HP', 'Type 1')
generate_boxplot(pokemon_data, 'Sp. Atk', 'Type 1')
generate_boxplot(pokemon_data, 'Sp. Def', 'Type 1')
generate_boxplot(pokemon_data, 'Attack', 'Legendary')
generate_boxplot(pokemon_data, 'Defense', 'Legendary')
generate_boxplot(pokemon_data, 'Speed', 'Legendary')
generate_boxplot(pokemon_data, 'Sp. Atk', 'Legendary')
generate_boxplot(pokemon_data, 'Sp. Def', 'Legendary')
pokemon_data[pokemon_data['Legendary'] == 1]['Type 1'].value_counts()
def clustered_scatterplot(x, y, cluster_column):
    return sns.scatterplot(data=pokemon_data, x=pokemon_data[x], y=pokemon_data[y], hue=cluster_column)
clustered_scatterplot('Total', 'Attack', 'Legendary')
clustered_scatterplot('Total', 'Defense', 'Legendary')
clustered_scatterplot('Total', 'Speed', 'Legendary')
clustered_scatterplot('Total', 'HP', 'Legendary')
clustered_scatterplot('Attack', 'Sp. Atk', 'Legendary')
clustered_scatterplot('Attack', 'Sp. Def', 'Legendary')
