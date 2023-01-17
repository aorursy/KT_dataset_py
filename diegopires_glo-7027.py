import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize

import os

import json

import re



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

sns.set()

%matplotlib inline
movie_df = pd. read_csv('../input/movierecommendationcompetition/movies.dat', sep='::', header=None, engine='python')

movie_df.rename(index=str, columns={0: "movieId", 1: "title", 2: "genre"}, inplace=True)

movie_df.dropna(inplace=True)

movie_df = movie_df.drop_duplicates(keep='first') # La doc dit qu'il y a des dupliques par movieId
user_df = pd. read_csv('../input/movierecommendationcompetition/users.dat', sep='::', header=None, engine='python')

user_df.rename(index=str, columns={0: "userId", 1: "gender", 2: "age", 3: "occupation", 4: "zipcode"}, inplace=True)

user_df.dropna(inplace=True)
rating_df = pd. read_csv('../input/movierecommendationcompetitionrating/training_ratings_for_kaggle_comp.csv')

rating_df.rename(index=str, columns={"user": "userId", "movie": "movieId"}, inplace=True)

rating_df.drop(["id"], axis=1, inplace=True)

rating_df.dropna(inplace=True)



df = pd.merge(movie_df, rating_df, on='movieId')

df = pd.merge(df, user_df, on='userId')

df.head()
df.columns
age_labels = { 1 : "Under 18", 18 : "18-24", 25 : "25-34", 35 : "35-44", 45: "45-49", 50 : "50-55", 56 : "56+" }



user_df["age_desc"] = user_df["age"]

user_df["age_desc"].replace(age_labels, inplace=True)



occupation_labels = {0:  "other or not specified",1:  "academic/educator",2:  "artist",3:  "clerical/admin",

                     4:  "college/grad student", 5:  "customer service",6:  "doctor/health care",

                     7:  "executive/managerial",8:  "farmer",9:  "homemaker", 10:  "K-12 student",

                     11:  "lawyer",12:  "programmer",13:  "retired",14:  "sales/marketing",15:  "scientist",

                     16:  "self-employed",17:  "technician/engineer",18:  "tradesman/craftsman",

                     19:  "unemployed",20:  "writer" }



user_df["occupation_desc"] = user_df["occupation"]

user_df["occupation_desc"].replace(occupation_labels, inplace=True)

    

gender_labels = { 'F' : "Woman", "M" : "Man"}

    

user_df["gender_desc"] = user_df["gender"]

user_df["gender_desc"].replace(gender_labels, inplace=True)



user_df.head()
import sys

!{sys.executable} -m pip install uszipcode
from uszipcode import SearchEngine

search = SearchEngine(simple_zipcode=True)



def get_zipcode_detail(row):                                                  

    zipcode = search.by_zipcode(row["zipcode"])

    return zipcode.major_city, zipcode.post_office_city, zipcode.county, zipcode.state



user_df['major_city'], user_df['post_office_city'], user_df["county"], user_df["state"]  = zip(*user_df.apply(get_zipcode_detail, axis=1))



user_df.head()
user_df.info()
gender_labels = { 'F' : "1", "M" : "2"}

    

user_df["gender_nr"] = user_df["gender"]

user_df["gender_nr"].replace(gender_labels, inplace=True)

user_df["gender_nr"] = user_df["gender_nr"].astype(float)



user_df.head()
cleaned_df = user_df.copy()



cleaned_df.dropna(inplace=True)

cleaned_df.drop_duplicates(inplace=True)



cleaned_df = cleaned_df.fillna("")

#cleaned_df.info()

sns.pairplot(cleaned_df, vars=["age", "occupation", "gender_nr"]);
f, ax = plt.subplots(figsize=(20, 20))

corr = user_df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
movie_df["year"] = movie_df.title.str.extract(r'\(([^)]*)\)[^(]*$', expand=True)

movie_df["title"].replace(regex=True,inplace=True,to_replace=r'\(([^)]*)\)[^(]*$',value=r'')

movie_df['title'] = movie_df['title'].str.strip()



movie_df.head()
# create new column with the count of genre

movie_df["genre_count"] = movie_df['genre'].apply(lambda x: len(x.split("|")))

max_genre_count = movie_df["genre_count"].max()



# create new dataframe with all genres

genres_df = movie_df["genre"].str.split("|", n = max_genre_count, expand = True) 



# merge it together

movie_df = pd.concat([movie_df, genres_df], axis=1)

movie_df.head()
movie_df.rename(index=str, columns={0: "genre_0", 1: "genre_1", 2: "genre_2", 3: "genre_3", 4: "genre_4", 5: "genre_5"}, inplace=True)



movie_df.head()
genres = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

for genre in genres:

    genre_name = genre.replace("'", "").replace("-","_")

    movie_df[genre_name] = [True if genre in x else False for x in movie_df['genre']]



movie_df.head()
movie_df.drop(["genre_1", "genre_2", "genre_3", "genre_4", "genre_5"], axis=1, inplace=True)

movie_df.rename(index=str, columns={"genre_0": "main_genre"}, inplace=True)

movie_df.head()
genre_df = movie_df[["Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary","Drama","Fantasy","Film_Noir","Horror","Musical","Mystery","Romance","Sci_Fi","Thriller","War","Western"]].copy()



genre_count = genre_df.apply(pd.value_counts)

genre_count.loc[True].plot.bar(figsize=(20,6), title="Frequency of genres")
movie_df.year.value_counts().plot(figsize=(20,6), kind='bar')

plt.title("Year")
f, ax = plt.subplots(figsize=(20, 20))

corr = movie_df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
df = pd.merge(movie_df, rating_df, on='movieId')

df = pd.merge(df, user_df, on='userId')

df.head()
df.info()
df.describe()
fig = plt.figure(figsize=(18,6), dpi=1600) 

alpha=alpha_scatterplot = 0.2 

alpha_bar_chart = 0.55



ax1 = plt.subplot2grid((2,3),(0,0))

user_df.gender_desc.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

ax1.set_xlim(-1, 2)

plt.title("Sexe")   

    

plt.subplot2grid((2,3),(0,1), colspan=2)

user_df.age_desc.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

ax1.set_xlim(-1, 2)

plt.title("Age")



plt.subplot2grid((2,3),(1,0), colspan=3)

user_df.occupation_desc.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

ax1.set_xlim(-1, 2)

plt.title("Occupation")

    
fig = plt.figure(figsize=(18,6))



df.age[df.rating == 1].plot(kind='kde')    

df.age[df.rating == 2].plot(kind='kde')

df.age[df.rating == 3].plot(kind='kde')

df.age[df.rating == 4].plot(kind='kde')

df.age[df.rating == 5].plot(kind='kde')

plt.xlabel("Age")    

plt.title("Rating Distribution within age")

plt.legend(('1*', '2*','3*', '4*', '5*'),loc='best') 
df.groupby('rating').size().sort_values(ascending=True).plot(kind='bar')
cleaned_df = df.copy()



cleaned_df.dropna(inplace=True)

cleaned_df.drop_duplicates(inplace=True)



# Visualize pairplot of df

#  sns.pairplot(cleaned_df, hue='rating');

cleaned_df=cleaned_df.fillna("")

sns.pairplot(cleaned_df, hue='rating');
tmdb_movie_df = pd. read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')



tmdb_cast_df = pd. read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

tmdb_cast_df.rename(index=str, columns={"movie_id": "id"}, inplace=True)

tmdb_cast_df.drop(["title"], axis=1, inplace=True)



tmdb_df = pd.merge(tmdb_movie_df, tmdb_cast_df, on='id')

tmdb_df.head()
tmdb_df.drop(['genres', 'homepage', 'id', 'overview', 'release_date', 'status', 'production_countries', 'tagline', 'spoken_languages', 'crew', 'runtime'], axis=1, inplace=True)

tmdb_df.head()
tmdb_df.columns
# This function transform a column with JSON data into multiple columns each with a value from the JSON

def parse_col_json_multiple_column(df, column, key):

    for index,i in zip(df.index,df[column].apply(json.loads)):

        for j in range(len(i)):

            df.loc[index, column + '.' + str(j)] = str((i[j][key]))

    tmdb_df.drop([key], axis=1, inplace=True)

    return df



# This function transform a column with JSON data into one column with all values separated by ','

def parse_col_json_one_column(df, column, key):

    for index,i in zip(df.index, df[column].apply(json.loads)):

        list1=[]

        for j in range(len(i)):

            list1.append((i[j][key])) 

        df.loc[index,column]= "|".join(list1)

    return df



def parse_col_json_one_column_first_value(df, column, key):

    for index,i in zip(df.index, df[column].apply(json.loads)):

        list1=[]

        if len(i) > 0:

            df.loc[index,column]= (i[0][key])

            df.loc[index,column + "_id"]= (i[0]["id"])

    return df

    

tmdb_df = parse_col_json_one_column_first_value(tmdb_df, 'cast', 'name')

tmdb_df = parse_col_json_one_column_first_value(tmdb_df, 'production_companies', 'name')

tmdb_df = parse_col_json_one_column(tmdb_df, 'keywords', 'name')

tmdb_df.head()
copy.describe()
full_df = pd.merge(df, tmdb_df, left_on='title', right_on='original_title', how='left')

full_df.head()
full_df.columns
df.info()
result = tmdb_df.query('original_title.str.contains("The ")', engine='python')



result.head()
result = movie_df.query('title.str.contains(", The")', engine='python')



result.head()
df.replace(regex=r'(, The$)', value='', inplace=True)



result = df.query('title.str.contains(", The")', engine='python')



result.head()

#result.info()
tmdb_df.replace(regex=r'(^The )', value='', inplace=True)



result = tmdb_df.query('title.str.contains("The ")', engine='python')



result.head()

#result.info()
full_df = pd.merge(df, tmdb_df, left_on='title', right_on='title', how='left')

full_df.head()
full_df.info()
f, ax = plt.subplots(figsize=(20, 20))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)