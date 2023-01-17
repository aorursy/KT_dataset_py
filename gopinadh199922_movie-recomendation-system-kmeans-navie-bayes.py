# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from itertools import chain



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

input_dir = "/kaggle/input/movielens-25m-grouplensorg"

input_dir = "/kaggle/input/movielens-mllatestsmall"

for dirname, _, filenames in os.walk(input_dir):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

chunk_size = 200000
### READING CSV FILES ###



# movies_dataframe = pd.read_csv(input_dir + "/ml-25m/movies.csv",header=0)

movies_dataframe = pd.read_csv(input_dir + "/ml-latest-small/movies.csv",header=0)

# movies_dataframe["title"] = movies_dataframe["title"].str.split("(").str[0].str.rstrip().str.split(" ")

movies_dataframe["genres"] = movies_dataframe["genres"].str.split("|")

# tags_dataframe = pd.read_csv(input_dir + "/ml-25m/tags.csv",header=0).drop(["timestamp","userId"],axis=1)

# genome_tags_dataframe = pd.read_csv(input_dir + "/ml-25m/genome-tags.csv",header=0)



# huge datasets reading

ratings_dataframe = []

# for chunk in pd.read_csv(input_dir + "/ml-25m/ratings.csv",header=0,chunksize=chunk_size):

for chunk in pd.read_csv(input_dir + "/ml-latest-small/ratings.csv",header=0,chunksize=chunk_size):

#     print(chunk.size)

    chunk = chunk.drop(["timestamp"],axis=1)

    ratings_dataframe.append(chunk)

ratings_dataframe = pd.concat(ratings_dataframe,axis=0)



# genome_scores_dataframe = []

# for chunk in pd.read_csv(input_dir + "/ml-25m/genome-scores.csv",header=0,chunksize=chunk_size):

# #     print(chunk.size)

#     genome_scores_dataframe.append(chunk)

# genome_scores_dataframe = pd.concat(genome_scores_dataframe,axis=0)



print("DONE READING")
print(movies_dataframe.shape)

print(movies_dataframe.head(1))

print("\n\n")

print(ratings_dataframe.shape)

print(ratings_dataframe.head(1))

print("\n\n")



# print(tags_dataframe.shape)

# print(tags_dataframe.head(1))

# print("\n\n")

# print(genome_tags_dataframe.shape)

# print(genome_tags_dataframe.head(1))

# print("\n\n")

# print(genome_scores_dataframe.shape)

# print(genome_scores_dataframe.head(1))

# print("\n\n")
ratings_dataframe = ratings_dataframe.groupby("movieId")

ratings_dataframe = pd.DataFrame({

    "movieId" : ratings_dataframe["movieId"],

    "score" : ratings_dataframe.count()["rating"],

    "rating" : ratings_dataframe.mean()["rating"]

})

ratings_dataframe = ratings_dataframe.drop("movieId",axis=1)

movies_dataframe = pd.merge(ratings_dataframe, movies_dataframe, on='movieId')

del ratings_dataframe



movies_dataframe
genres_lens = movies_dataframe['genres'].map(len)

movies_dataframe = pd.DataFrame({

    'movieId': np.repeat(movies_dataframe['movieId'], genres_lens),

    'score': np.repeat(movies_dataframe['score'], genres_lens),

    'rating': np.repeat(movies_dataframe['rating'], genres_lens),

    'title': np.repeat(movies_dataframe['title'], genres_lens),

    'genre': chain.from_iterable(movies_dataframe['genres'])

})

movies_dataframe.head(1)
# title_lens = movies_dataframe['title'].map(len)

# movies_dataframe = pd.DataFrame({'movieId': np.repeat(movies_dataframe['movieId'], title_lens),

#               'rating': np.repeat(movies_dataframe['rating'], title_lens),

#               'genre': np.repeat(movies_dataframe['genre'], title_lens),

#               'title': chain.from_iterable(movies_dataframe['title'])

#              })

# movies_dataframe.head(1)


# tags = pd.merge(genome_scores_dataframe, genome_tags_dataframe, on='tagId').drop(["tagId"],axis=1)

# del genome_scores_dataframe,genome_tags_dataframe



# tags_dataframe = pd.merge(tags, tags_dataframe,  how='left', left_on=['movieId','tag'], right_on = ['movieId','tag'])

# del tags



# tags_dataframe
genre_rating_avg = movies_dataframe.groupby("genre")["rating"].mean()

genre_rating_avg = pd.DataFrame({'genre':genre_rating_avg.index, 'genre_rating':genre_rating_avg.values})



movies_dataframe = pd.merge(movies_dataframe,genre_rating_avg,on="genre")



movies_dataframe
# genres_title = movies_dataframe.drop(["score","rating","genre_rating"],axis=1)

# movies_dataframe = movies_dataframe.drop(["genre","title"],axis=1)



# movies_dataframe
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=100).fit(movies_dataframe.drop(["movieId","genre","title"],axis=1))

centroids = kmeans.cluster_centers_

# print(centroids)

cluster_map = pd.DataFrame()

# cluster_map['data_index'] = movies_dataframe.index.values

cluster_map['category'] = kmeans.labels_

cluster_map
movies_dataframe = pd.merge(movies_dataframe,cluster_map,left_index=True,right_index=True)

movies_dataframe
fig = plt.figure(figsize=(20,10))

graph = fig.add_axes([0.1,0.1,0.8,0.8])

graph.scatter(movies_dataframe['movieId'], movies_dataframe['rating'], c= kmeans.labels_.astype(float), s=1, alpha=0.1)

graph.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.show()
fig = plt.figure(figsize=(20,10))

graph = fig.add_axes([0.1,0.1,0.8,0.8])

graph.scatter(movies_dataframe['movieId'], movies_dataframe['genre_rating'], c= kmeans.labels_.astype(float), s=1, alpha=0.1)

graph.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.show()
fig = plt.figure(figsize=(20,10))

graph = fig.add_axes([0.1,0.1,0.8,0.8])

graph.scatter(movies_dataframe['movieId'], movies_dataframe['score'], c= kmeans.labels_.astype(float), s=1, alpha=0.1)

graph.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

plt.show()
movies_dataframe.to_csv("movies.csv", mode='w',index=False, header=True)

# tags_dataframe.to_csv('tags.csv', mode='w',index=False, header=True)

print("DONE WRITING")
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(movies_dataframe.drop(["movieId","title","genre","category"],axis=1), movies_dataframe["category"])
test = pd.DataFrame({

    "movieId" : np.array([8644,32031,136800,140174,140627,141818,141994]),

    "title" : np.array(["I, Robot (2004)","Robots (2005)","Robot Overlords (2014)","Room (2015)","Battle For Sevastopol (2015)","Ordinary Miracle (1978)","Saving Christmas (2014)"]),

    "genres" : np.array(["Action|Adventure|Sci-Fi|Thriller","Adventure|Animation|Children|Comedy|Fantasy|Sci-Fi|IMAX","Action|Adventure|Sci-Fi","Drama","Drama|Romance|War","Comedy|Drama|Fantasy|Romance","Children|Comedy"])

})

test
test["genres"] = test["genres"].str.split("|")

genres_lens = test['genres'].map(len)

test = pd.DataFrame({

    'movieId': np.repeat(test['movieId'], genres_lens),

    'title': np.repeat(test['title'], genres_lens),

    'genre': chain.from_iterable(test['genres'])

})



test
most_frequent_genre = test.groupby("genre").count()["title"].max()

most_frequent_genre = test.groupby("genre").count().loc[test.groupby("genre").count()["title"] == most_frequent_genre]

most_frequent_genre
df = movies_dataframe.loc[movies_dataframe['genre'].isin(most_frequent_genre.index)]



genre_rating = df.groupby("genre")["rating"].mean()

genre_rating = pd.DataFrame({

    "genre" : genre_rating.index,

    "genre_rating" :  genre_rating.values

})



df = pd.merge(df.drop("genre_rating",axis=1),genre_rating,on="genre")

df["score"] = df["score"] + 1

df
category = model.predict(df.drop(["movieId","title","genre","category"],axis=1))

df["category"] = category



df
frequent_category = df.groupby("category").count()["title"]

frequent_category = frequent_category.loc[frequent_category.max() == frequent_category.values].index



df = df.loc[df['category'].isin(frequent_category.values)]

df = df.loc[~df['movieId'].isin(test["movieId"])]



df = df.drop_duplicates(subset=['movieId',"title","score","rating"])



df
top100_watched = df.nlargest(100,["score"])

top10_rated = top100_watched.nlargest(10,["rating"])



top10_rated
# from lightfm.datasets import fetch_movielens

# from lightfm import LightFM



# #fetch data from model

# data = fetch_movielens(min_rating = 4.0)



# #create model

# model = LightFM(loss = 'warp')



# #train mode

# model.fit(data['train'], epochs=30, num_threads=2)



# #recommender fucntion

# def sample_recommendation(model, data, user_ids):

#     #number of users and movies in training data

#     n_users, n_items = data['train'].shape

#     print(n_items)

#     for user_id in user_ids:

#     	#movies they already like

#         known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

#         #movies our model predicts they will like

#         scores = model.predict(user_id, np.arange(n_items))

#         #sort them in order of most liked to least

#         top_items = data['item_labels'][np.argsort(-scores)]

#         #print out the results

#         print("User %s" % user_id)

#         print("     Known positives:")



#         for x in known_positives[:3]:

#             print("        %s" % x)



#         print("     Recommended:")



#         for x in top_items[:3]:

#             print("        %s" % x)

            

# sample_recommendation(model, data, [3, 25, 451])