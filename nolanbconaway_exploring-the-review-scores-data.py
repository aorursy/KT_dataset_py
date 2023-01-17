import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.signal import savgol_filter



pd.set_option('precision', 2)

np.set_printoptions(precision=2)



con = sqlite3.connect('../input/database.sqlite')

reviews = pd.read_sql('SELECT * FROM reviews', con)

genres = pd.read_sql('SELECT * FROM genres', con)

con.close()
print('\nAverages:')

print(np.mean(reviews[['best_new_music', 'score']]))



print('\nStandard Deviation:')

print(np.std(reviews[['best_new_music', 'score']]))



g = reviews.groupby('score')

info = g['best_new_music'].agg(['sum','count']).reset_index()



plt.plot(info['score'], savgol_filter(info['count'], 5, 1), label = 'All Reviews') 

plt.plot(info['score'], savgol_filter(info['sum'], 5, 1), label = "Best New Music") 

plt.legend(loc = 'best')

plt.xlabel('Score')

plt.ylabel('Frequency')

plt.show()
idx = (reviews.best_new_music == 0) & (reviews.score == 10.0)

reviews.loc[idx, ['artist', 'title', 'pub_date'] ]
genre_data = pd.merge(reviews[['reviewid','score']], genres, 

                  on = 'reviewid')



g = genre_data.groupby('genre')

table = g['score'].agg(['count', 'mean', 'std']).reset_index()



# plot the average at each level of count

avgline = table.groupby('count')['mean'].mean().reset_index()

avgline['mean'] = savgol_filter(avgline['mean'], 5, 1)

plt.plot(avgline['count'], avgline['mean'],'k--')



plt.plot(table['count'],table['mean'],'o', alpha = 1)



for j, row in table.iterrows():

    curr_avg = float(avgline.loc[avgline['count'] == row['count'], 'mean'])

    jitter = np.random.uniform(0.1, high = 0.5)

    if row['mean'] < curr_avg: jitter*= -1.0

    

    plt.plot([row['count'], row['count']], [row['mean'], row['mean'] + jitter], 'k-', alpha = 0.1)

    plt.text(row['count'], row['mean'] + jitter, row['genre'], 

             ha = 'center', va = 'center')



plt.ylabel('Average Score')

plt.xlabel('Number of Reviews')

plt.ylim([5, 10])

plt.show()
g = reviews.groupby('author')

table = g.score.agg(('mean','std','count'))

table['ratio'] = table['mean'] / table['count']



# remove labels with only a handful of reviews

table = table.loc[table['count'] > 15]



# plot the average at each level of count

avgline = table.groupby('count')['mean'].mean().reset_index()

avgline['mean'] = savgol_filter(avgline['mean'], 5, 1)

plt.plot(avgline['count'], avgline['mean'],'k--')



# plot each author as a point

plt.plot(table['count'], table['mean'],'o', alpha = 0.5)



# identify some standouts

items = [

         table['mean'].idxmax(), 

         table['mean'].idxmin(),

         table['count'].idxmax()

        ]



for idx in items:

    

    x, y = table.loc[idx, 'count'], table.loc[idx, 'mean']

    curr_avg = float(avgline.loc[avgline['count'] == x, 'mean'])

        

    jitter = np.random.uniform(0.1, high = 0.5)

    if y < curr_avg: jitter*= -1.0

    

    plt.plot([x, x], [y, y + jitter], 'k-', alpha = 0.1)

    plt.text(x, y + jitter, idx, ha = 'center', va = 'center')



plt.ylabel('Average Score')

plt.xlabel('Number of Reviews')

plt.ylim([5, 10])

plt.show()