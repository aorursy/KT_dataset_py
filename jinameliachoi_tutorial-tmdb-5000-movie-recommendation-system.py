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
import warnings

warnings.filterwarnings('ignore')



movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

print(movies.shape)

movies.head(3)
df = movies[['id', 'title', 'genres', 'vote_average',

            'vote_count', 'popularity', 'keywords', 'overview']]
# 일부 칼럼은 파이썬 리스트 내부에 여러 개의 딕셔너리가 있는 형태로 표현되어 있습니다

pd.set_option('max_colwidth', 100)

df[['genres', 'keywords']][:1]
# 칼럼의 문자열을 분해해 개별 장르를 파이썬 리스트로 만듭니다

from ast import literal_eval

df['genres'] = df['genres'].apply(literal_eval)

df['keywords'] = df['keywords'].apply(literal_eval)
# name 값만 리스트 객체로 변환

df['genres'] = df['genres'].apply(lambda x: [y['name'] for y in x])

df['keywords'] = df['keywords'].apply(lambda x: [y['name'] for y in x])

df[['genres', 'keywords']][:1]
from sklearn.feature_extraction.text import CountVectorizer



df['genres_literal'] = df['genres'].apply(lambda x: (' ').join(x))

count_vect = CountVectorizer(min_df=0,

                            ngram_range=(1,2))

genre_mat = count_vect.fit_transform(df['genres_literal'])

print(genre_mat.shape)
# 코사인 유사도 계산하기

from sklearn.metrics.pairwise import cosine_similarity



genre_sim = cosine_similarity(genre_mat, genre_mat)

print(genre_sim.shape)

print(genre_sim[:1])
# genre_sim 객체의 기준 행별로 비교 대상이 되는 행의 유사도 값이 높은 순으로 정렬된 행렬의 위치 인덱스 값을 추출

genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]

print(genre_sim_sorted_ind[:1])
# 레코드별 장르 코사인 유사도 인덱스를 가지고 있는 df, 영화제목, 건수를 입력하면 추천 영화 정보를 가지는 df 반환

def find_sim_movie(df, sorted_ind, title_name, top_n=10):

    # 인자로 입력된 df에서 'title' 칼럼이 입력된 title_name 값인 dataframe 추출

    title_movie = df[df['title'] == title_name]

    

    # title_name을 가진 df의 index를 ndarray로 반환하고

    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n개의 index 추출

    title_index = title_movie.index.values

    similar_indexes = sorted_ind[title_index, :(top_n)]

    

    # 추출된 top_n index 출력. top_n index는 2차원 데이터임

    # df에서 index로 사용하기 위해 1차원 array로 변경

    print(similar_indexes)

    similar_indexes = similar_indexes.reshape(-1)

    

    return df.iloc[similar_indexes]
similar_movies = find_sim_movie(df,

                               genre_sim_sorted_ind,

                               'The Godfather',

                               10)

similar_movies[['title', 'vote_average']]
# 이번에는 좀 더 많은 후보군을 선정한 뒤에 영화의 평점에 따라 필터링하여 최종 추전하는 방식

# 주의할 점은, `vote_average`는 소수의 관객이 특정 영화에 만적이나 높은 평점을 부여해 왜곡된 데이터를 가지고 있음

df[['title', 'vote_average', 'vote_count']].sort_values('vote_average',

                                                       ascending=False)[:10]
C = df['vote_average'].mean()

m = df['vote_count'].quantile(0.6)

print('C:', round(C, 3), 'm: ', round(m, 3))
# 새로운 평점 정보 만들기

percentile = 0.6

m = df['vote_count'].quantile(percentile)

C = df['vote_average'].mean()



def weighted_vote_average(record):

    v = record['vote_count']

    R = record['vote_average']

    

    return ((v / (v+m)) * R) + ((m / (m+v)) * C)



df['weighted_vote'] = df.apply(weighted_vote_average, axis=1)
df[['title', 'vote_average', 'weighted_vote', 'vote_count']].sort_values('weighted_vote', ascending=False)[:10]
# 새롭게 정의된 평점 기준에 따라 영화를 추천합니다

def find_sim_movie(df, sorted_ind, title_name, top_n=10):

    title_movie = df[df['title'] == title_name]

    title_index = title_movie.index.values

    

    # top_n에 2배에 해당하는 장르 유사성이 높은 인덱스 추출

    similar_indexes = sorted_ind[title_index, :(top_n*2)]

    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 인덱스는 제외

    similar_indexes = similar_indexes[similar_indexes != title_index]

    

    # top_n의 2배에 해당하는 후보군에서 weighted_vote가 높은 순으로 top_n만큼 추출

    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]



similar_movies = find_sim_movie(df,

                               genre_sim_sorted_ind,

                               'The Godfather', 10)

similar_movies[['title', 'vote_average', 'weighted_vote']]