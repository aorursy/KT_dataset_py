import numpy   as np

import pandas  as pd

import seaborn as sns



import plotly.express as px
!ls ../input/anime-recommendations-database
# Read csv files

animes_df  = pd.read_csv('../input/anime-recommendations-database/anime.csv')

ratings_df = pd.read_csv('../input/anime-recommendations-database/rating.csv')
# Rename column, display sample

#animes_df.columns = ['anime_id', 'name', 'genre', 'type', 'episodes', 'average_rating', 'members']

animes_df.rename({'rating':'average_rating'}, axis=1, inplace=True)

animes_df.sample(5).style.background_gradient(cmap=sns.light_palette("green", as_cmap=True))
# Inspect amount of rows and columns

print(animes_df.shape)

print(ratings_df.shape)
# List, tuple and dict

a = [123, 456]

b = (123, 456)

c = {'key':'value'}
# Inspect Data types

ratings_df.dtypes
# Turn into string so that colormap is ignored

ratings_df['user_id']  = ratings_df['user_id'].astype('str')

ratings_df['anime_id'] = ratings_df['anime_id'].astype('str')

animes_df['anime_id']  = animes_df['anime_id'].astype('str')
# Inspect ratings df

ratings_df.sample(5).style.background_gradient(cmap=sns.light_palette("green", as_cmap=True)).hide_index()
len(animes_df['name'].unique())
# Join both dataframes

full_df = pd.merge(ratings_df, animes_df, on='anime_id', how='left')

full_df
fig = px.histogram(full_df,

                 x       = 'average_rating',

                 nbins   = 100,

                 opacity = 0.8,

                 title   = 'Histogram of Average Rating',

                 template = 'plotly_dark',

                 color_discrete_sequence=['#23c6cc','#23c6cc']

                )





fig.update_xaxes(title='Average Rating')

fig.update_yaxes(title='Count')

fig.show()
full_df.head()
fig = px.box(animes_df,

       x = 'type',

       y = 'average_rating',

      )



fig.update_xaxes(title='Type')

fig.update_yaxes(title='Average Rating')

fig.show()
fig = px.scatter(fus_df_df,

             y       = 'average_rating',

             x       = 'episodes',

             opacity = 0.8,

             title   = 'Histogram of Average Rating',

             template  = 'plotly_dark',

            )





fig.update_xaxes(range=[0,500])

fig.show(ani)

full_df.query("name == 'Naruto' and rating > -1")['name']