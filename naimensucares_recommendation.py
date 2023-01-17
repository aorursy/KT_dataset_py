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
rating=pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")
movie=pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
# genome_score=pd.read_csv("/kaggle/input/movielens-20m-dataset/genome_scores.csv")
# genome_tag=pd.read_csv("/kaggle/input/movielens-20m-dataset/genome_tags.csv")
# tag=pd.read_csv("/kaggle/input/movielens-20m-dataset/tag.csv")


print(movie.info())
print()
print(rating.info())

movie.head(3)
rating.head(3)


df=movie.merge(rating)
df.head()
df=df.drop("timestamp",axis=1)
data = df.iloc[1000000:]
display(data.sample(3))


def Title(data,title):
    
    tit=data["title"].str.startswith(title[:8])
    tit=data[tit]
    return tit["title"].unique()
title=Title(data,'Halloween (1978)')
title
def rate_cor(data,title):
    pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
    
    movie_watched = pivot_table[title]
    
    similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
    similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
    return similarity_with_other_movies.head()
rate=rate_cor(data,'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)')
rate

def genres(data,title):

    gen=data[data["title"]==title]
    genres=gen["genres"].unique()[0]
    print(genres)
    gen_data=data["genres"].str.contains(genres)
    gen_data=data[gen_data]
    df=gen_data["genres"].str.startswith(genres[:len(genres)//2])
    gen_data=gen_data[df]
    gen_data=gen_data.groupby("title")["rating"].mean()
    gen_data=gen_data.sort_values(ascending=False)
    return gen_data
   
    
    
    
    
 
genres=genres(data,"Flintstones, The (1994)")

genres
# # s=['Mystery', 'Fantasy', 'IMAX', 'Horror', 'Crime', 'Sci-Fi', 'Documentary', 'Children', 'Action', 'War', 'Musical', 'Comedy', 'Adventure', 'Drama', 'Thriller', 'Romance',
#  'Animation']