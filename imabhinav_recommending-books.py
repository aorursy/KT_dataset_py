!pip install isbnlib

!pip install newspaper3k

!pip install goodreads_api_client


import numpy as np 

import pandas as pd

import os

import seaborn as sns

import isbnlib

from newspaper import Article

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from tqdm import tqdm

from progressbar import ProgressBar

import re

from scipy.cluster.vq import kmeans, vq

from pylab import plot, show

from matplotlib.lines import Line2D

import matplotlib.colors as mcolors

import goodreads_api_client as gr

from sklearn.cluster import KMeans

from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/books.csv', error_bad_lines = False)
df.index = df['bookID']
#Finding Number of rows and columns

print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))
df.head()
df.replace(to_replace='J.K. Rowling-Mary GrandPré', value = 'J.K. Rowling', inplace=True)
df.head()
#Taking the first 20:



sns.set_context('poster')

plt.figure(figsize=(20,15))

books = df['title'].value_counts()[:20]

rating = df.average_rating[:20]

sns.barplot(x = books, y = books.index, palette='deep')

plt.title("Most Occurring Books")

plt.xlabel("Number of occurances")

plt.ylabel("Books")

plt.show()
most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['ratings_count'], most_rated.index, palette='rocket')





sns.set_context('talk')

most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('authors')

plt.figure(figsize=(15,10))

ax = sns.barplot(most_books['title'], most_books.index, palette='icefire_r')

ax.set_title("Top 10 authors with most books")

ax.set_xlabel("Total number of books")

for i in ax.patches:

    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')



#Finding the top 15 authors with the most number of books

df['authors'].value_counts().head(10)
df.average_rating.isnull().value_counts()
df.dropna(0, inplace=True)

#Removing Any null values
plt.figure(figsize=(10,10))

rating= df.average_rating.astype(float)

sns.distplot(rating, bins=20)



trial = df[['average_rating', 'ratings_count']]

data = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count'])]).T



X = data

distortions = []

for k in range(2,30):

    k_means = KMeans(n_clusters = k)

    k_means.fit(X)

    distortions.append(k_means.inertia_)



fig = plt.figure(figsize=(15,10))

plt.plot(range(2,30), distortions, 'bx-')

plt.title("Elbow Curve")
#Computing K means with K = 5, thus, taking it as 5 clusters

centroids, _ = kmeans(data, 5)



#assigning each sample to a cluster

#Vector Quantisation:



idx, _ = vq(data, centroids)
# some plotting using numpy's logical indexing

sns.set_context('paper')

plt.figure(figsize=(15,10))

plt.plot(data[idx==0,0],data[idx==0,1],'or',#red circles

     data[idx==1,0],data[idx==1,1],'ob',#blue circles

     data[idx==2,0],data[idx==2,1],'oy', #yellow circles

     data[idx==3,0],data[idx==3,1],'om', #magenta circles

     data[idx==4,0],data[idx==4,1],'ok',#black circles

    

     

        

        

        

        

        )

plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8, )









circle1 = Line2D(range(1), range(1), color = 'red', linewidth = 0, marker= 'o', markerfacecolor='red')

circle2 = Line2D(range(1), range(1), color = 'blue', linewidth = 0,marker= 'o', markerfacecolor='blue')

circle3 = Line2D(range(1), range(1), color = 'yellow',linewidth=0,  marker= 'o', markerfacecolor='yellow')

circle4 = Line2D(range(1), range(1), color = 'magenta', linewidth=0,marker= 'o', markerfacecolor='magenta')

circle5 = Line2D(range(1), range(1), color = 'black', linewidth = 0,marker= 'o', markerfacecolor='black')



plt.legend((circle1, circle2, circle3, circle4, circle5)

           , ('Cluster 1','Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'), numpoints = 1, loc = 0, )





plt.show()
trial.idxmax()
trial.drop(3, inplace = True)

trial.drop(41865, inplace = True)
data = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count'])]).T





#Computing K means with K = 8, thus, taking it as 8 clusters

centroids, _ = kmeans(data, 5)



#assigning each sample to a cluster

#Vector Quantisation:



idx, _ = vq(data, centroids)
# some plotting using numpy's logical indexing

sns.set_context('paper')

plt.figure(figsize=(15,10))

plt.plot(data[idx==0,0],data[idx==0,1],'or',#red circles

     data[idx==1,0],data[idx==1,1],'ob',#blue circles

     data[idx==2,0],data[idx==2,1],'oy', #yellow circles

     data[idx==3,0],data[idx==3,1],'om', #magenta circles

     data[idx==4,0],data[idx==4,1],'ok',#black circles

    

     

        

        

        

        

        )

plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8, )









circle1 = Line2D(range(1), range(1), color = 'red', linewidth = 0, marker= 'o', markerfacecolor='red')

circle2 = Line2D(range(1), range(1), color = 'blue', linewidth = 0,marker= 'o', markerfacecolor='blue')

circle3 = Line2D(range(1), range(1), color = 'yellow',linewidth=0,  marker= 'o', markerfacecolor='yellow')

circle4 = Line2D(range(1), range(1), color = 'magenta', linewidth=0,marker= 'o', markerfacecolor='magenta')

circle5 = Line2D(range(1), range(1), color = 'black', linewidth = 0,marker= 'o', markerfacecolor='black')



plt.legend((circle1, circle2, circle3, circle4, circle5)

           , ('Cluster 1','Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'), numpoints = 1, loc = 0, )





plt.show()
df['Ratings_Dist'] = segregation(df)

books_features = pd.concat([df['Ratings_Dist'].str.get_dummies(sep=","), df['average_rating'], df['ratings_count']], axis=1)
books_features.head()
min_max_scaler = MinMaxScaler()

books_features = min_max_scaler.fit_transform(books_features)
np.round(books_features, 2)
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')

model.fit(books_features)

distance, indices = model.kneighbors(books_features)



def get_index_from_name(name):

    return df[df["title"]==name].index.tolist()[0]



all_books_names = list(df.title.values)



def get_id_from_partial_name(partial):

    for name in all_books_names:

        if partial in name:

            print(name,all_books_names.index(name))

            

def print_similar_books(query=None,id=None):

    if id:

        for id in indices[id][1:]:

            print(df.iloc[id]["title"])

    if query:

        found_id = get_index_from_name(query)

        for id in indices[found_id][1:]:

            print(df.iloc[id]["title"])
print_similar_books("The Catcher in the Rye")

print_similar_books("The Hobbit or There and Back Again")
get_id_from_partial_name("Harry Potter and the ")
print_similar_books(id = 1) #ID for the Book 5