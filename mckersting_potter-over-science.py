import pandas as pd



df = pd.read_csv('../input/books.csv', error_bad_lines=False)

df.head()
df.tail()
df.sort_values(by=['average_rating', 'ratings_count'], ascending=False)

df2 = df.loc[df['ratings_count']>100]

df.sort_values(by=['average_rating', 'ratings_count'], ascending=False)

df2.head()
df2.tail()
df2.loc[df['authors'] == 'Neil deGrasse Tyson']
df2.loc[df['authors'] == 'Richard Dawkins']
df2.loc[df['authors'] == 'Stephen Hawking']
df2.loc[df['authors'] == 'Carl Sagan']
science_writers = 'Neil deGrasse Tyson', 'Richard Dawkins', 'Stephen Hawking', 'Carl Sagan'

harry_potter = 'J.K. Rowling-Mary GrandPré', 'J.K. Rowling'
science_df = df[df['authors'].isin(science_writers)]

science_df.head()
harry_df = df[df['authors'].isin(harry_potter)]

harry_df.head()
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")

sns.set_context("talk")
harry_stde = harry_df['average_rating'].sem()

science_stde = science_df['average_rating'].sem()

ax = sns.barplot(['Harry Potter','Science'], [harry_df['average_rating'].mean(), science_df['average_rating'].mean()], yerr=[harry_stde, science_stde], palette=("Blues_d"))

for bar in ax.patches:

    x = bar.get_x()

    width = bar.get_width()

    center = x + width/2.0



    bar.set_x(center - 0.6/ 2.0)

    bar.set_width(.6)
from scipy.stats import ttest_ind

[tstat, p] = ttest_ind(harry_df['average_rating'], science_df['average_rating'])

p