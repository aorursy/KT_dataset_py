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
#reading the dataset of jokes
joke=pd.read_csv('/kaggle/input/jester-collaborative-filtering-dataset/JokeText.csv')
joke.head(10)
#reading the review dataset
review=pd.read_csv('/kaggle/input/jester-collaborative-filtering-dataset/UserRatings1.csv')
review.head(3)
#finding the shape of the jokes dataset
joke.shape
joke.JokeId.unique()
review2=pd.read_csv('/kaggle/input/jester-collaborative-filtering-dataset/UserRatings2.csv')
review2.head(3)
#finding the 2 nd joke in the dataset
joke[joke['JokeId']==2]
#displaying the second joke
joke.JokeText[joke['JokeId']==2]
#filling the NAN values with 0
review=review.fillna(0)
review.head(3)
#setting joke id as the index
review=review.set_index('JokeId')
#transposing the review dataset to get the users at the index as the users' ratings are going to be compared here
rating=review.T
rating.head(3)
#normalizing the value of each row
def standardize(row):
    new_row=(row-row.mean())/(row.max()-row.min())
    return new_row
rating_new=rating.apply(standardize)
rating_new.tail(5)
#finding the cosine similarities among the user ratings
from sklearn.metrics.pairwise import cosine_similarity
similar_jokes=cosine_similarity(rating_new.T)
print(similar_jokes)
#making the cosine similarity result into a new datast with the joke id on both axis
#this helps us to undestand the similarity between the jokes 
ratings_df=pd.DataFrame(similar_jokes,index=rating_new.columns,columns=rating_new.columns)
ratings_df.head(5)
d=rating.describe()
d=d.T
d.describe()
# this function calculates the similarity score or the angular distance between the cosine similarities of the ratings(rating-mean rating)
# it displays the joke ids with similar scores
def jokes_to_recommend(jokeid,rating):
    similar_score=ratings_df[jokeid]*(rating-0.64)
    similar_score=similar_score.sort_values(ascending=False)
    return similar_score
print(jokes_to_recommend(1,1))
Joke_rater=[(0,9),(1,9),(1,1)]
similar_jokes=pd.DataFrame()
for jokes,rating in Joke_rater:
    similar_jokes=similar_jokes.append(jokes_to_recommend(jokes,rating),ignore_index=True)
similar_jokes.head()
similar_jokes.sum().sort_values(ascending=False)
s=[]
t=[]
j=joke.JokeText[joke['JokeId']==2]
print(j)
print("How much will you rate this joke out of 20")
n1=int(input())
t.append([2,n1])
j2=joke.JokeText[joke['JokeId']==3]
print(j2)
print("How much will you rate this joke out of 20")
n2=int(input())
t.append([3,n2])
j3=joke.JokeText[joke['JokeId']==4]
print(j3)
print("How much will you rate this joke out of 20")
n3=int(input())
t.append([4,n3])
'''def find_jokes(t):
    similar_jokes=pd.DataFrame()
    new_set=np.array()
    for jokes,rating in t:
        similar_jokes=similar_jokes.append(jokes_to_recommend(jokes,rating),ignore_index=True)
    new_set=similar_jokes.sum().sort_values(ascending=False)
    print(new_set)
find_jokes(t)'''
#Joke recommender function
def find_jokes():
    t=[]
    remove_str1=["\n",'[',']']
    for i in range(2,5):
        a=joke.JokeText[joke['JokeId']==i].values
        for j in a:
            if j not in remove_str1:
                print(j)
        print("How much will you rate this joke out of 20")
        n1=int(input())
        t.append([i,n1])
        print("\n")
    a=""
    similar_jokes=pd.DataFrame()
    new_set=[]
    for jokes,rating in t:
        similar_jokes=similar_jokes.append(jokes_to_recommend(jokes,rating),ignore_index=True)
    new_set=similar_jokes.sum().sort_values(ascending=False).index
    #print(len(new_set))
    remove_str=["\n",'[',']']
    for i in range(1,6):
        a=joke.JokeText[joke['JokeId']==(new_set[i])].values
        for j in a:
            if j not in remove_str:
                print(j)
        print("\n")
find_jokes()
