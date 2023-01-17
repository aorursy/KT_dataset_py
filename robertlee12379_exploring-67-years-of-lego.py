# Nothing to do here
# Import modules

import pandas as pd



# Read colors data

colors = pd.read_csv('../input/colors.csv')



# Print the first few rows

colors.head()
# How many distinct colors are available?

num_colors = colors.rgb.count()

print(num_colors)
# colors_summary: Distribution of colors based on transparency

colors_summary = colors.groupby(colors.is_trans).count()

colors_summary
%matplotlib inline

# Read sets data as `sets`

sets = pd.read_csv('../input/sets.csv')

# Create a summary of average number of parts by year: `parts_by_year`

parts_by_year = sets[['year','num_parts']].groupby('year',as_index=False).mean()

# Plot trends in average number of parts by year

parts_by_year.plot(x='year',y='num_parts')
# themes_by_year: Number of themes shipped by year

themes_by_year = sets[['year','theme_id']].groupby('year',as_index=False).agg({"theme_id": pd.Series.count})

themes_by_year.head(2)
# Nothing to do here