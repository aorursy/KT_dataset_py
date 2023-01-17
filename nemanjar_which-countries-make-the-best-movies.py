# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sb



matplotlib.style.use('fivethirtyeight')



sb.set

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movie_metadata.csv')

print(movies.info())
all_genres = sorted(list(set('|'.join(movies.genres).split('|'))))

print(all_genres)
genres_dummy = pd.DataFrame(np.zeros([movies.shape[0], len(all_genres)]),

                            columns=all_genres)



genre_lists = movies.genres.apply(lambda x: x.split('|'))



for c in genres_dummy.columns:

    genres_dummy[c] = genre_lists.apply(lambda x: c in x)



print('Sanity check, row 1000:')    

print(genre_lists[999])

print(genres_dummy.ix[999,:])
movies = pd.concat([movies, genre_lists, genres_dummy], axis=1).drop('genres', axis=1)
plt.hist(movies.num_critic_for_reviews.dropna(), 50);

plt.title('Histogram of number of critics per movie')
plt.figure()

plt.subplot(211)

plt.hist(movies.imdb_score.dropna());

plt.subplot(212)

plt.hist([b for b in movies.budget if b < 1e8], 30);



movies_filtered = movies.copy()
score_by_cntry = movies_filtered.groupby('country',

                                         as_index=False).mean().sort_values(by='imdb_score',

                                                                            ascending=False)

count_by_cntry = movies_filtered.groupby('country',

                                         as_index=False).count().sort_values(by='duration',

                                                                             ascending=False)

count_by_cntry['count'] = count_by_cntry['duration']





sb.factorplot(data=movies_filtered, y='country', x='imdb_score', 

              kind='bar', order=score_by_cntry.country, size=20, aspect=0.5)



plt.title('Mean IMDb score per country')



sb.factorplot(data=count_by_cntry, y='country', x='count', 

              kind='bar', order=count_by_cntry.country, size=20, aspect=0.5)



plt.title('Number of movies in the dataset - PER COUNTRY');
score_and_count = score_by_cntry.merge(count_by_cntry.ix[:, ['country','count']],

                                        on='country')



plt.figure(figsize=[10,6])

plt.scatter(score_and_count['count'], score_and_count.imdb_score, s=200, alpha=0.6)

plt.xscale('log')

plt.xlim([0.1, 1e4])

plt.title('Average IMDB score per country vs. Number of movies from that country');