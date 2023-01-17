import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('precision', 2)

np.set_printoptions(precision=2)



con = sqlite3.connect('../input/database.sqlite')

all_reviews = pd.read_sql('SELECT * FROM reviews', con)

con.close()
author_groups = all_reviews.groupby('author')

author_counts = author_groups.size().reset_index()



x = range(120)

y = [sum(author_counts[0] > i) for i in x]

plt.bar(x, y, width = 1, edgecolor='none')

plt.xlabel('Number of Reviews')

plt.ylabel('Authors with N+ Reviews')

plt.show()
# subset data

keep_authors = author_counts.author[author_counts[0] >= 70]

reviews = all_reviews[all_reviews.author.isin(keep_authors)]



# clear out space for review order

reviews = reviews.assign(review_num = pd.Series(index=reviews.index))



# count each author's reviews

author_groups = reviews.groupby('author')

for a, rows in author_groups:

    values = list(reversed(range(rows.shape[0])))

    reviews.set_value(rows.index, 'review_num', values)

    
g = reviews.groupby(['review_num'])

table = g['score'].agg(['mean','std', 'count'])

table = table.reset_index()



x = table['review_num']

y = table['mean']



plt.fill_between(x, y + table['std'], y  - table['std'], alpha = 0.5, label = '1 SD')    

plt.plot(x, y, '-', color = 'k', label = 'Mean')

plt.legend()

plt.axis([0, 200, 0, 10])

plt.xlabel('Review Number')

plt.ylabel('Score')

plt.show()
# subset data

keep_authors = author_counts.author[author_counts[0] < 70]

reviews = all_reviews[all_reviews.author.isin(keep_authors)]



# clear out space for review order

reviews = reviews.assign(review_num = pd.Series(index=reviews.index))



# count each author's reviews

author_groups = reviews.groupby('author')

for a, rows in author_groups:

    values = list(reversed(range(rows.shape[0])))

    reviews.set_value(rows.index, 'review_num', values)





# plotting

g = reviews.groupby(['review_num'])

table = g['score'].agg(['mean','std', 'count'])

table = table.reset_index()



x = table['review_num']

y = table['mean']



plt.fill_between(x, y + table['std'], y  - table['std'], alpha = 0.5, label = '1 SD')    

plt.plot(x, y, '-', color = 'k', label = 'Mean')

plt.legend()

plt.axis([0, 70, 0, 10])

plt.xlabel('Review Number')

plt.ylabel('Score')

plt.show()