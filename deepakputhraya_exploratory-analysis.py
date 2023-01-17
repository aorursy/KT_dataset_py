# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.display import display, HTML # embed html inside ipython

from bokeh.charts import Bar, show, output_notebook # charts

output_notebook()



# Anime dataframe containing general information about the anime

anime_df = pd.read_csv('../input/anime.csv', header=0)

# Rating by users

rating_df = pd.read_csv('../input/rating.csv', header=0)
# Fill genre with NA

anime_df.genre.fillna('NA', inplace=True)

# Get a list of all genre

genres = set((', '.join(anime_df.genre.values.flatten())).split(', '))
display(genres)
def has_genre(genre, genre_list):

    if genre in genre_list.split(', '):

        return 1

    return 0



genre_count = {

    'genre' : [],

    'count' : []

}



# Make each genre as a feature

for genre in genres:

    anime_df[genre] = anime_df.apply(lambda row: has_genre(genre, row['genre']), axis=1)

    genre_count['genre'].append(genre)

    genre_count['count'].append(anime_df[genre][anime_df[genre] == 1].count())
bar = Bar(genre_count, values='count', label=['genre'],

           agg='mean', title="Genre Distribution")



show(bar)