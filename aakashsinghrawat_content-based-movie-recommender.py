# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
data = pd.read_csv('/kaggle/input/imdb-5000-movie-dataset/movie_metadata.csv')
data.head()
data.shape, data.columns
# keep important features which can help to recommend movies, drop others
data = data.drop(['color', 'num_critic_for_reviews', 'duration',
        'director_facebook_likes', 'actor_3_facebook_likes',
        'actor_1_facebook_likes', 'gross', 
        'num_voted_users', 'cast_total_facebook_likes',
        'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link',
        'num_user_for_reviews', 'language', 'country',
        'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes',
        'imdb_score', 'aspect_ratio', 'movie_facebook_likes'], 1)
data.head()
data.info()
data.dropna(inplace=True)
## clean genres--- remove | between generes
data['genres'] = data['genres'].apply(lambda a: str(a).replace('|', ' '))
data['genres']
data['movie_title'][0]
data['movie_title'] = data['movie_title'].apply(lambda a:a[:-1])
data['movie_title'][0]
## combined features on which we will calculate cosine similarity
data['combined'] = data['director_name']+' '+data['actor_2_name']+' '+data['genres']+' '+data['actor_1_name']+' '+data['actor_3_name']
data.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vec = CountVectorizer()
vec_matrix = vec.fit_transform(data['combined'])
similarity = cosine_similarity(vec_matrix)
def recommend(movie):
    if movie not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==movie].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        plt.bar(l, [i[1] for i in lst])
        plt.xticks(rotation=90)
        plt.xlabel('similar movies to---> '+movie)
        plt.ylabel('cosine scores')
        return l
data['movie_title'].sample(10)
recommend('The Kids Are All Right')
recommend('The Dark Knight Rises')
recommend('Pirates of the Caribbean: At World\'s End')
