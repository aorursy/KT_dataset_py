import pandas as pd
df=pd.read_csv("../input/movie-dataset/movie_dataset.csv")
df.columns
df.head()
features=["director","spoken_languages","keywords","genres"]

for f in features:
    df[f]=df[f].fillna(" ")
def func(row):
    return row["keywords"]+" "+row["genres"]+" "+row["spoken_languages"]+row["director"]
df["additional features"]=df.apply(func,axis=1)
df.head()
df["additional features"]
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["additional features"])
print(count_matrix)
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix=cosine_similarity(count_matrix)
similarity_matrix
def get_index(movie_name):
    for i in range(0,df.shape[0]):
        if(df["original_title"][i]==movie_name):
            return df["index"][i]
def get_movies(array):
    
    movies=[]
    i=0;
    for index in array[1:len(array)]:
        movies.append(df["original_title"][index[0]])
        i+=1
        if(i==10):
            break;
    return movies

        

    
movie_userliked=input()
import numpy as np
print("Enter Movie You Liked")

movie_index=get_index(movie_userliked)
recommendedindex=list(enumerate(similarity_matrix[movie_index]))
recommendedindex=sorted(recommendedindex,key=lambda x:x[1],reverse=True)
recommendedmovies=get_movies(recommendedindex)
print("Recommended Movies because you liked "+" "+movie_userliked)
recommendedmovies=np.array(recommendedmovies)
print(recommendedmovies.reshape(len(recommendedmovies),1))


