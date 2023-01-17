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
df = pd.read_csv("../input/imdb-extensive-dataset/IMDb movies.csv")
df.columns
df = df[['title','genre', 'year', 'duration', 'avg_vote', 'votes']]

df.head()
df.info()
df["year"].unique()
df = df.dropna()
df["year"].unique()
df["genre"].unique()
df["votes"].unique()
df["year"]= df["year"].str.replace("TV Movie 2019", "2019", case = False) 

  
df["year"] = df["year"].astype(float)
movies = df.copy()
df["year"].unique()
movies = df
def build_chart(df,percentile=0.9):
    
    print("Enter your preferred Genre type")
    genre= input()
    print("Enter the minimum length of the movie")
    low_length = int(input())
    print("Enter the maximum length of the movie")
    high_length = int(input())
    print("Enter the earliest year  of the movie")
    low_year =int(input())
    print("Enter the latest year of the movie")
    high_year =int(input())
    movies = df.copy()
    movies = movies[(movies['genre']==genre) &(movies['duration']>=low_length) &(movies['duration']<=high_length)&(movies['year']>=low_year)&(movies['year']<=high_year)]
    c = movies['avg_vote']
    m = movies['votes'].quantile(percentile)
    
    #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    q_movies = movies.copy().loc[movies['votes'] >= m]
    
    #Calculate score using the IMDB formula
    q_movies['score']= q_movies['avg_vote']
    q_movies['score'] = q_movies.apply(lambda x: (x['votes']/(x['votes']+m) * x['avg_vote'])+ (m/(m+x['votes']) * c),axis=1)

    #Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)
    
    return q_movies
movies
def weighted_rating(x, m=m, C=c):
    v = x['votes']
    R = x['avg_vote']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies = build_chart(df,0.8)
q_movies
q_movies
len(q_movies)

