# Pandas for managing datasets

import pandas as pd
# Matplotlib for additional customization



from matplotlib import pyplot as plt



%matplotlib inline
# install seaborn



!pip install seaborn
# Seaborn for plotting and styling

import seaborn as sns
# Read dataset

Pokemon_df = pd.read_csv('../input/Pokemon_2.csv')
# Display first 5 observations

Pokemon_df.head()
# Recommended way to draw a Scatter Plot from a Dataframe



sns.lmplot(x='Attack', y='Defense', data=Pokemon_df)
# Scatterplot arguments

sns.lmplot(x='Attack', y='Defense', 

           data=Pokemon_df,

           fit_reg=False, # No regression line

           hue='Stage')   # Color by evolution stage
# Scatterplot arguments

sns.lmplot(x='Attack', y='Defense', 

           data=Pokemon_df,

           fit_reg=False, # No regression line

           hue='Stage')   # Color by evolution stage



# Tweak using Matplotlib

plt.title('Scatter Plot using Seaborn Package')
# Boxplot



plt.figure(figsize=(14,7))



sns.boxplot(data=Pokemon_df)



# Set theme



plt.figure(figsize=(14,7))



sns.set_style('whitegrid')

 

# Violin plot

sns.violinplot(x='Stage', y='Attack', data=Pokemon_df)
# Swarm plot with Pokemon color palette



pkmn_type_colors = ['#78C850',  # Stage 1 

                    '#F08030',  # Stage 2

                    '#6890F0'   # Stage 3

                   ]



plt.figure(figsize=(14,7))



sns.swarmplot(x='Stage', y='Attack', data=Pokemon_df, 

              palette=pkmn_type_colors)
# Set figure size with matplotlib

plt.figure(figsize=(10,6))

 

# Create plot

sns.violinplot(x='Stage',

               y='Attack', 

               data=Pokemon_df, 

               inner=None, # Remove the bars inside the violins

               palette=pkmn_type_colors)

 

sns.swarmplot(x='Stage', 

              y='Attack', 

              data=Pokemon_df, 

              color='k', # Make points black

              alpha=0.7) # and slightly transparent

 

# Set title with matplotlib

plt.title('Attack by Stage')
# Calculate correlations

corr = Pokemon_df.corr()



plt.figure(figsize=(14,7))



# Heatmap

sns.heatmap(corr)
# Distribution Plot (a.k.a. Histogram)



plt.figure(figsize=(14,7))



sns.distplot(Pokemon_df.Attack)

# Count Plot (a.k.a. Bar Plot)



plt.figure(figsize=(14,7))



sns.countplot(x='Stage', 

              data=Pokemon_df, 

              palette=pkmn_type_colors)

 

# Rotate x-labels

plt.xticks(rotation=-45)