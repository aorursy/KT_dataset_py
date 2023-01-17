import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
alt = 'https://raw.githubusercontent.com/premonish/Springboard/master/Data/tmdb_5000_movies.csv'

movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies.columns
credits.columns
#credits.cast.value_counts()
print('Mean budget:', str(round(movies.budget.mean(),2)))

print('Median budget:', str(movies.budget.median()))

print('Budget Mode:', str(movies.budget.mode()[0]))
print('Mean budget:', str(round(movies.revenue.mean(),2)))

print('Median budget:', str(movies.revenue.median()))

print('Budget Mode:', str(movies.revenue.mode()[0]))
print('Mean budget:', str(round(movies.runtime.mean(),2)))

print('Median budget:', str(movies.runtime.median()))

print('Budget Mode:', str(movies.runtime.mode()[0]))
new = movies.genres.value_counts()

new.head()
#movies.keywords.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
co2 = np.corrcoef((movies.budget, movies.revenue))[0][1]
sns.set_style("darkgrid")

sns.regplot(x='budget', y='revenue', data=movies)

plt.title('Pearson\'s CorrCoef: '+str(co2))
ax = sns.kdeplot(movies.revenue, shade=True, color="b")
ax = sns.kdeplot(movies.budget, shade=True, color="b")
sns.regplot(x='runtime', y='revenue', data=movies)
co3 = np.corrcoef((movies.popularity, movies.revenue))[0][1]

sns.regplot(x='popularity', y='revenue', data=movies)

print(co3)
sns.regplot(x='vote_average', y='revenue', data=movies)
# create a feature 'pct' ||| revenue : budget ratio (higher is better)

movies['pct'] = movies.revenue/movies.budget

movies.pct = movies.pct.replace(np.inf, '0')

movies.pct.fillna(0)

movies['pct'].head()

#print(movies['pct'].mean(axis = 1), skipna = True)

movies.pct = movies.pct.astype(float)

h = movies.sort_values(by=['pct'], ascending=False)

h2 = h[['original_title', 'pct']]
h.original_title.head(100)
h2.head(20)
fig, ax = plt.subplots()



plt.rcParams['font.family'] = "serif"

ax.barh(h2.original_title[:2], h2.pct[:2], align='center')

ax.set_yticks(h2.original_title[:2])

ax.set_yticklabels(h2.original_title[:2])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Revenue/Budget')

ax.set_title('Films with Outsized Returns')



plt.show()
fig, ax = plt.subplots()



plt.rcParams['font.family'] = "serif"

ax.barh(h2.original_title[2:5], h2.pct[2:5], align='center')

ax.set_yticks(h2.original_title[2:5])

ax.set_yticklabels(h2.original_title[2:5])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Revenue/Budget')

ax.set_title('Films with Outsized Returns')



plt.show()
fig, ax = plt.subplots()



plt.rcParams['font.family'] = "serif"

ax.barh(h2.original_title[5:25], h2.pct[5:25], align='center')

ax.set_yticks(h2.original_title[5:25])

#ax.set_xlabels(h2.pct[5:25])

ax.set_yticklabels(h2.original_title[5:25])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Revenue/Budget')

ax.set_title('Films with Outsized Returns')



plt.show()
ax = sns.kdeplot(movies.pct, shade=True, color="b")
sns.pairplot(movies)
co4 = np.corrcoef((movies.vote_count, movies.revenue))[0][1]

co4
movies2 = movies[['budget','popularity','revenue','runtime','vote_average','vote_count']]
ax = sns.pairplot(movies2)