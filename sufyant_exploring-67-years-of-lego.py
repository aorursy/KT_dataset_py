# Import modules

import pandas as pd



# Read colors data

colors = pd.read_csv('../input/lego-database/colors.csv')



# Print the first few rows

colors.head()
# How many distinct colors are available?

num_colors = len(colors.name.unique())

num_colors
# colors_summary: Distribution of colors based on transparency

colors_summary = colors.groupby('is_trans').count()

print(colors_summary)
%matplotlib inline

import matplotlib.pyplot as plt

# Read sets data as `sets`

sets = pd.read_csv('../input/lego-database/sets.csv')

# Create a summary of average number of parts by year: `parts_by_year`

parts_by_year = sets.groupby('year').mean

# Plot trends in average number of parts by year

fig, ax = plt.subplots()

ax.plot(parts_by_year().index, parts_by_year().num_parts)



ax.set(xlabel='Years', ylabel='Number of Parts')

ax.grid()

plt.show()

# themes_by_year: Number of themes shipped by year

import numpy as np

themes_by_year = sets.groupby('year')[['theme_id']].count()

themes_by_year['year'] = themes_by_year.index

themes_by_year.index = np.arange(0, len(themes_by_year))

themes_by_year