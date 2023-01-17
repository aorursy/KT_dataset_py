# Pandas and numpy for data manipulation

import pandas as pd

import numpy as np
# matplotlib for plotting

import matplotlib.pyplot as plt

import matplotlib



# Set text size

matplotlib.rcParams['font.size'] = 18



# Seaborn for pairplots

import seaborn as sns



sns.set_context('talk', font_scale=1.2);
df = pd.read_csv('../input/gapminder-data/gapminder_data.csv')

df.columns = ['country', 'continent', 'year', 'life_exp', 'pop', 'gdp_per_cap']

df.head()
df.describe()
sns.pairplot(df);
df['log_pop'] = np.log10(df['pop'])

df['log_gdp_per_cap'] = np.log10(df['gdp_per_cap'])



df = df.drop(columns = ['pop', 'gdp_per_cap'])
matplotlib.rcParams['font.size'] = 40

sns.pairplot(df, hue = 'continent');
sns.pairplot(df, hue = 'continent', diag_kind = 'kde', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, size = 4);
df['decade'] = pd.cut(df['year'], bins = range(1950, 2010, 10))

df.head()
sns.pairplot(df, hue = 'decade', diag_kind = 'kde', vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'],

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, size = 4);
sns.pairplot(df[df['year'] >= 2000], vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], 

             hue = 'continent', diag_kind = 'kde', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, size = 4);

plt.suptitle('Pair Plot of Socioeconomic Data for 2000-2007', size = 28);
# Create an instance of the PairGrid class.

grid = sns.PairGrid(data= df[df['year'] == 2007],

                    vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], size = 4)



# Map different plots to different sections

grid = grid.map_upper(plt.scatter, color = 'darkred')

grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')

grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', edgecolor = 'k');
# Function to calculate correlation coefficient between two arrays

def corr(x, y, **kwargs):

    

    # Calculate the value

    coef = np.corrcoef(x, y)[0][1]

    # Make the label

    label = r'$\rho$ = ' + str(round(coef, 2))

    

    # Add the label to the plot

    ax = plt.gca()

    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)

    

# Create a pair grid instance

grid = sns.PairGrid(data= df[df['year'] == 2007],

                    vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], size = 4)



# Map the plots to the locations

grid = grid.map_upper(plt.scatter, color = 'darkred')

grid = grid.map_upper(corr)

grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')

grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred');
# Define a summary function

def summary(x, **kwargs):

    # Convert to a pandas series

    x = pd.Series(x)

    

    # Get stats for the series

    label = x.describe()[['mean', 'std', 'min', '50%', 'max']]

    

    # Convert from log to regular scale

    # Adjust the column names for presentation

    if label.name == 'log_pop':

        label = 10 ** label

        label.name = 'pop stats'

    elif label.name == 'log_gdp_per_cap':

        label = 10 ** label

        label.name = 'gdp_per_cap stats'

    else:

        label.name = 'life_exp stats'

       

    # Round the labels for presentation

    label = label.round()

    ax = plt.gca()

    ax.set_axis_off()

    print(label)

    # Add the labels to the plot

    #ax.annotate(pd.DataFrame(label),xy = (0.1, 0.2), size = 20, xycoords = ax.transAxes)    

    



# Create a pair grid instance

grid = sns.PairGrid(data= df[df['year'] == 2007],

                    vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], size = 4)



# Fill in the mappings

grid = grid.map_upper(plt.scatter, color = 'darkred')

grid = grid.map_upper(corr)

grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')

grid = grid.map_diag(summary);