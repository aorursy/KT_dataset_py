# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
imdb_df = pd.read_csv('../input/movie_metadata.csv')

pd.set_option('display.max_columns', None)

imdb_df.head()
cols = list(imdb_df.columns.values)

cols
df = imdb_df[['movie_title','title_year','duration','genres','imdb_score','num_voted_users',

              'movie_facebook_likes','director_name','director_facebook_likes','actor_1_name',

              'actor_1_facebook_likes','actor_2_name','actor_2_facebook_likes',

              'actor_3_name','actor_3_facebook_likes','cast_total_facebook_likes',

              'num_critic_for_reviews','num_user_for_reviews','gross','budget',

              'language','country','content_rating','aspect_ratio','plot_keywords',

              'color','facenumber_in_poster','movie_imdb_link']]
df.head()
df.tail()
df.loc[:, imdb_df.isnull().any()].head()
(df.info())
df.describe().round()
df.describe(include=['O'])
df.sort_values('actor_1_facebook_likes', ascending=False).head(10)

#df.sort_values('imdb_score', ascending=False).head(10)

#df.sort_values('movie_facebook_likes', ascending=False).head(10)

#df.sort_values('cast_total_facebook_likes', ascending=False).head(10)

#df.sort_values('actor_1_facebook_likes', ascending=False).head(10).actor_1_name.unique()