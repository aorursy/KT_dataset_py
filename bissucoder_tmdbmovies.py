# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import json
import re
from datetime import datetime
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credits_df=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")
movies_df=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
credits_df.head()
movies_df.head()
movies_df.isna().sum()
movies_df['release_date'].isna().sum()
movies_df.dropna(subset=['release_date'],inplace=True)
movies_df['release_date'].isna().sum()
movies_df[movies_df['overview'].isna()]
movies_df.dropna(subset=['overview'],inplace=True)
movies_df.isna().sum()
movies_df[movies_df['tagline'].isna()].head()
credits_df.isna().sum()
all_movies_df=pd.merge(left=movies_df,right=credits_df,left_on='id',right_on='movie_id',
                      suffixes=('_left','_right'))
all_movies_df.drop(['id','title_left'],axis=1,inplace=True)
all_movies_df=all_movies_df.rename(columns={'title_right':"title"})
all_movies_df = all_movies_df[['movie_id', 'budget', 'title', 'original_title', 'status', 'tagline', 'release_date', 'runtime', 
               'genres', 'production_companies', 'production_countries', 'popularity', 'revenue', 'vote_average',
               'vote_count', 'cast', 'crew', 'homepage', 'keywords', 'original_language', 'overview', 'spoken_languages'
             ]]
allm = all_movies_df.copy()


json_columns=['genres', 'keywords', 'production_countries', 'spoken_languages']
for column in json_columns:
    all_movies_df[column] = all_movies_df[column].apply(json.loads, encoding="utf-8")

seperate_cols_array_json=["cast","crew","production_companies"]
for columns in seperate_cols_array_json:
    all_movies_df[columns]=all_movies_df[columns].apply(json.loads,encoding="utf-8")
    
    
cast_df=pd.concat([all_movies_df["movie_id"],all_movies_df['cast']],axis=1)
crew_df=pd.concat([all_movies_df["movie_id"],all_movies_df['crew']],axis=1)
production_companies_df=pd.concat([all_movies_df["movie_id"],all_movies_df['production_companies']],axis=1)
keywords_df=pd.concat([all_movies_df["movie_id"],all_movies_df['keywords']],axis=1)
genres_df=pd.concat([all_movies_df["movie_id"],all_movies_df['genres']],axis=1)
production_countries_df=pd.concat([all_movies_df["movie_id"],all_movies_df['production_countries']],axis=1)
spoken_languages_df=pd.concat([all_movies_df["movie_id"],all_movies_df['spoken_languages']],axis=1)
cast_columns=["cast_id","character","credit_id","gender","id","name","order"]
crew_columns=["credit_id","department","gender","id","job","name"]
prod_columns=["name","id"]
languages_columns=["iso_639_1","name"]
production_countries_columns=["iso_3166_1","name"]


def process_cast(df):
    rows_list=[]
    for index,x  in zip(cast_df.index,cast_df["cast"]):
        for i in range(len(x)):
            data={}
            for column_name in cast_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=cast_df.loc[index,"movie_id"]
            rows_list.append(data)
    cast_df_new=pd.DataFrame(rows_list)
    return cast_df_new

def process_production_companies(df):
    rows_list=[]
    for index,x  in zip(production_companies_df.index,production_companies_df["production_companies"]):
        for i in range(len(x)):
            data={}
            for column_name in prod_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=crew_df.loc[index,"movie_id"]
            rows_list.append(data)
    df=pd.DataFrame(rows_list)
    return df

def process_keywords(df):
    rows_list=[]
    for index,x  in zip(keywords_df.index,keywords_df["keywords"]):
        for i in range(len(x)):
            data={}
            for column_name in prod_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=crew_df.loc[index,"movie_id"]
            rows_list.append(data)
    df=pd.DataFrame(rows_list)
    return df

def process_spoken_languages(df):
    rows_list=[]
    for index,x  in zip(spoken_languages_df.index,spoken_languages_df["spoken_languages"]):
        for i in range(len(x)):
            data={}
            for column_name in languages_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=crew_df.loc[index,"movie_id"]
            rows_list.append(data)
    df=pd.DataFrame(rows_list)
    return df

def process_genres(df):
    rows_list=[]
    for index,x  in zip(genres_df.index,genres_df["genres"]):
        for i in range(len(x)):
            data={}
            for column_name in prod_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=crew_df.loc[index,"movie_id"]
            rows_list.append(data)
    df=pd.DataFrame(rows_list)
    return df

def process_crew(df):
    rows_list=[]
    for index,x  in zip(crew_df.index,crew_df["crew"]):
        for i in range(len(x)):
            data={}
            for column_name in crew_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=crew_df.loc[index,"movie_id"]
            rows_list.append(data)
    crew_df_new=pd.DataFrame(rows_list)
    return crew_df_new

def process_production_countries(df):
    rows_list=[]
    for index,x  in zip(production_countries_df.index,production_countries_df["production_countries"]):
        for i in range(len(x)):
            data={}
            for column_name in production_countries_columns:
                data[column_name]=x[i][column_name]
    #         print(len(data))
            data["movie_id"]=crew_df.loc[index,"movie_id"]
            rows_list.append(data)
    df=pd.DataFrame(rows_list)
    return df

cast_df_new=process_cast(all_movies_df)

crew_df_new=process_crew(all_movies_df)
production_df_companies_new=process_production_companies(all_movies_df)
key_df_new=process_keywords(all_movies_df)
genres_df_new=process_genres(all_movies_df)
spoken_df_new=process_spoken_languages(all_movies_df)
production_countries_df_new=process_production_countries(all_movies_df)

allmovienew=all_movies_df.copy()
#Drop cols
all_movies_df.drop(["genres","production_companies","production_countries","keywords","spoken_languages","cast","crew"],axis=1,inplace=True)
genres_df_new.head()
all_movies_df.head()
# def genre_comparisons():
genre=pd.merge(left=all_movies_df,right=genres_df_new,left_on='movie_id',right_on='movie_id',
                      suffixes=('_left','_right'))
#name here refers to genre name
genre_budget_group_by=genre.groupby('name').agg({'budget':'sum'})
#     pass
plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
plt.title("Count of Movies by Genre")
sns.countplot(data=genre,x="name")
genre_total_votes_group_by=genre.groupby('name').agg({'vote_count':'sum'})
plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
plt.title("Total Vote Count By Genre")
sns.barplot(data=genre_total_votes_group_by,x=genre_total_votes_group_by.index,y="vote_count")
genre_avg_votes_group_by=genre.groupby('name').agg({'vote_average':'sum'})
plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
plt.title("Average Votes By Genre")
sns.barplot(data=genre_avg_votes_group_by,x=genre_avg_votes_group_by.index,y="vote_average")
genre.corr()
genre_revenue_group_by=genre.groupby('name').agg({'revenue':'sum'})
plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
plt.title("Revenue By Genre")
sns.barplot(data=genre_revenue_group_by,x=genre_revenue_group_by.index,y="revenue")
all_movies_df['release_date']=pd.to_datetime(all_movies_df['release_date'])
all_movies_df['release_year']=all_movies_df['release_date'].dt.year
all_movies_df['release_month']=all_movies_df['release_date'].dt.month
all_movies_df.head(2)
after_date_time=all_movies_df.copy()
genre.head()
genre_popularity_group_by=genre.groupby('name').agg({'popularity':'mean'})
plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
plt.title("Popularity By Genre")
sns.barplot(data=genre_popularity_group_by,x=genre_popularity_group_by.index,y="popularity")
genre=pd.merge(left=all_movies_df,right=genres_df_new,left_on='movie_id',right_on='movie_id',
                      suffixes=('_left','_right'))
#name here refers to genre name
genre.dtypes
genre.head()
genre_names=genre['name'].unique()

g = sns.FacetGrid(genre, col="name",col_wrap=2)
g.map(sns.lineplot, "release_year", "popularity")
g.add_legend();
spoken_df_new.head()
all_movies_df.head()
#Join
spoken_df=pd.merge(left=all_movies_df,right=spoken_df_new,left_on='movie_id',right_on='movie_id',
                      suffixes=('_left','_right'))
spoken_df.head()
"""This method returns the top 10 languages based on the number of movies they are used in"""
def most_popular_languages():
    counts=spoken_df.groupby('name').agg({'movie_id':'count'})
    top_10_languages=counts.nlargest(10,"movie_id")
    top_10_languages.rename(columns={'movie_id':'count'},inplace=True)
    top_10_languages=top_10_languages.reset_index()
    return top_10_languages

    
top_10_languages=most_popular_languages()
top_10_languages
fig,ax=plt.subplots(1,2)
plt.tight_layout()
plt.subplot_tool()
plt.xlabel("Movie Languages and Budget Allocations")
sns.barplot(x="name",y="revenue",data=spoken_df[spoken_df['name'].isin(top_10_languages['name'])],ax=ax[0])
sns.barplot(x="name",y="budget",data=spoken_df[spoken_df['name'].isin(top_10_languages['name'])],ax=ax[1])
for tick in ax[0].get_xticklabels():
    tick.set_rotation(90)
for tick in ax[1].get_xticklabels():
    tick.set_rotation(90)
