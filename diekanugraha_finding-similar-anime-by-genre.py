# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

import collections

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_similarity_score # Jaccard Similarity
animes = pd.read_csv('../input/anime.csv') # load the data

animes['genre'] = animes['genre'].fillna('None') # filling 'empty' data

animes['genre'] = animes['genre'].apply(lambda x: x.split(', ')) # split genre into list of individual genre



genre_data = itertools.chain(*animes['genre'].values.tolist()) # flatten the list

genre_counter = collections.Counter(genre_data)

genres = pd.DataFrame.from_dict(genre_counter, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})

genres.sort_values('count', ascending=False, inplace=True)



# Plot genre

f, ax = plt.subplots(figsize=(8, 12))

sns.set_color_codes("pastel")

sns.set_style("white")

sns.barplot(x="count", y="genre", data=genres, color='b')

ax.set(ylabel='Genre',xlabel="Anime Count")
genre_map = {genre: idx for idx, genre in enumerate(genre_counter.keys())}

def extract_feature(genre):

    feature = np.zeros(len(genre_map.keys()), dtype=int)

    feature[[genre_map[idx] for idx in genre]] += 1

    return feature

    

anime_feature = pd.concat([animes['name'], animes['genre']], axis=1)

anime_feature['genre'] = anime_feature['genre'].apply(lambda x: extract_feature(x))

print(anime_feature.head(80))
test_data = anime_feature.take([0, 19, 1, 2, 16, 23, 6, 49, 220, 66])

for row in test_data.iterrows():

    print('Similar anime like {}:'.format(row[1]['name']))

    search = anime_feature.drop([row[0]]) # drop current anime

    search['result'] = search['genre'].apply(lambda x: jaccard_similarity_score(row[1]['genre'], x))

    search_result = search.sort_values('result', ascending=False)['name'].head(25)

    for res in search_result.values:

        print('\t{}'.format(res))

    print()