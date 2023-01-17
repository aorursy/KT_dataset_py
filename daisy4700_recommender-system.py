# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
d1=pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')
d1.head(3)
d2=pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
d2.head(3)
d2.columns
d=d2
d.head(3)
#dropping unwanted columns
d=d.drop(columns=['homepage','overview','production_countries','runtime','budget','revenue'])
d.head(3)
#extracting a data
extract=d.genres.loc[0]
type(extract)
#preprocessing data to prepare it for further use
a=[]
def run(d):
    st=["id","name",'"','"',':',',','{',"}",'0','1','2','3','4','5','6','7','8','9','[',']',' ']
    b=""
    for i in range(len(d.genres)):
        b=""
        e=d.genres.loc[i]
        for j in st:
            e=e.replace(j,'')
        a.append(e)
run(d)
#print(a)
d.genres=a
d.head(6)
a=[]
def run(d):
    st=["id","name",'"','"',':',',','{',"}",'0','1','2','3','4','5','6','7','8','9','[',']',' ']
    b=""
    for i in range(len(d.keywords)):
        b=""
        e=d.keywords.loc[i]
        for j in st:
            e=e.replace(j,'')
        a.append(e)
run(d)
#print(a)
d.keywords=a
d.head(6)
a=[]
def run(d):
    st=["id","name",'"','"',':',',','{',"}",'0','1','2','3','4','5','6','7','8','9','[',']',' ']
    b=""
    for i in range(len(d.production_companies)):
        b=""
        e=d.production_companies.loc[i]
        for j in st:
            e=e.replace(j,'')
        a.append(e)
run(d)
#print(a)
d.production_companies=a
d.head(3)
a=[]
def run(d):
    st=["id","name",'"','"',':',',','{',"}",'0','1','2','3','4','5','6','7','8','9','[',']',' ']
    b=""
    for i in range(len(d.spoken_languages)):
        b=""
        e=d.spoken_languages.loc[i]
        for j in st:
            e=e.replace(j,'')
        a.append(e)
run(d)
#print(a)
d.spoken_languages=a
d.head(6)
#looking at the data in the original_languages feature
d.original_language.unique()
d.head(3)
d.status.unique()
#dropping the row with no release date
check=pd.isnull(d['release_date'])
d[check]
d=d.drop(4553,axis=0)
d.head(3)
#dropping the tagline column
check=pd.isnull(d['tagline'])
d[check]
d=d.drop('tagline',axis=1)
d.head(3)
#present shape of the dataset
d.shape
d.popularity.max()
d.popularity.min()
#looking at the maximum and minimum value of popularity feature
d.describe()
#dropping the rows whoes popularity is below 30
dp=d.drop(d[d.popularity<30].index,axis=0)
dp.shape
dp=dp.drop('spoken_languages',axis=1)
dp.head(3)


#setting new index column
index=[]
for i in range(len(dp.genres)):
    index.append(i)
dp['Index']=index
dp.head(5)
#len(index)

dp.set_index('Index',inplace=True)
dp.head(3)
#dividing the release_date feature into year, month, date
year=[]
month=[]
date=[]
def convert_date(dp):
    for i in range(len(dp.release_date)):
        a=dp.release_date.loc[i]
        b=""
        c=""
        for j in range(len(a)):
            if(a[j]!='-'):
                b=b+a[j]
            else:
                break
        year.append(int(b))
        for k in range(5,len(a)):
            if(a[k]!='-'):
                c=c+a[k]
            else:
                break
        month.append(int(c))
        d=""
        for l in range(8,len(a)):
            if(a[l]!='-'):
                d=d+a[l]
            else:
                break
        date.append(int(d))
        
convert_date(dp)
dp['release_year']=year
dp['release_month']=month
dp['release_date']=date
dp.head(3)
dp.popularity=dp['popularity'].astype(int)
dp.head(3)
dp.status.unique()
dp.shape
dp=dp.drop(columns=['status'])
dp.head(3)
d1.head(3)
d1=d1.rename(columns={'movie_id':'id'})

#d1.set_index('id',inplace=True)
d1.head(3)
d1.columns
#result=pd.merge(dp,d1[['cast']],on='id')
'''cast1=[]
for i in dp.id:
    for j in range(len(d1.id)):
        if(i==d1.id.loc[j]):
            cast1.append(d1.cast.loc[j])
print(cast1)
        '''
#dp['cast']=cast1
dp.head(3)

#df=dp.iloc[:,:-1]
df=dp
df.head(3)
#making a different column that will contain the details of the popularity of the film 
df.vote_average=df.vote_average.astype(str)
df.popularity=df.popularity.astype(str)
df.release_year=df.release_year.astype(str)
def feature_join(row):
    return row['vote_average']+" "+row['popularity']+" "+row['release_year']+" "+row['original_language']
df["recent_popularity"]=df.apply(feature_join,axis=1)
df["recent_popularity"].head(3)
features=['keywords','genres','original_language']
#making a separate column that will contain all the basic features of the film
def feature_join(row):
    return row['keywords']+" "+row['genres']+" "+row['original_language']+" "+row['popularity']
df["all_feature"]=df.apply(feature_join,axis=1)
df["all_feature"].head(3)
#using CountVectorizer and cosine similarity to find the percentage of similarity between the films
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#two functions to find title from index and index from title
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title].index.values[0]
cv=CountVectorizer()
#finding the distance in vectors
count_matrix=cv.fit_transform(df.recent_popularity)
# finding the cosine similarity in a matrix form
similarity_scores=cosine_similarity(count_matrix)
user_movie="Man of Steel"
#finding index of the title given
movie_index=get_index_from_title(user_movie)
#forming a list of tuples containg the cosine values and the particular movie index
similar_movies=list(enumerate(similarity_scores[movie_index]))
#sorting in descending order of the cosine_similariities
sorted_similar=sorted(similar_movies,key=lambda x:x[1],reverse=True)
#printing the 10 most similar movies
i=0
for movie in sorted_similar:
    print(get_title_from_index(movie[0]))
    i=i+1
    if(i>10):
        break
#function that recommends the user the best film according to his likes  
def we_recommend_the_best_movies_you_like():
    print("Enter your favourite movie name:")
    movie=input()
    print("Enter 1 to get recent popular films else 2 to get some films that you would like")
    choice=input()
    cv=CountVectorizer()
    if choice==1:
        count_matrix=cv.fit_transform(df.recent_popularity)
    else:
        count_matrix=cv.fit_transform(df.all_feature)
    similarity_scores=cosine_similarity(count_matrix)
    user_movie=movie
    movie_index=get_index_from_title(user_movie)
    similar_movies=list(enumerate(similarity_scores[movie_index]))
    sorted_similar=sorted(similar_movies,key=lambda x:x[1],reverse=True)
    i=0
    for movie in sorted_similar:
        print(get_title_from_index(movie[0]))
        i=i+1
        if(i>10):
            break
we_recommend_the_best_movies_you_like()
