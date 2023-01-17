# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cosine

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ratings = pd.read_csv("../input/rating.csv")

movies = pd.read_csv("../input/movie.csv")



movies= movies.loc[:,["movieId","title"]]

#print(movies.head(10)) #27277



ratings = ratings.loc[:,["userId","movieId","rating"]]

#print(ratings.head(10))



# then merge movie and rating data

data = pd.merge(movies,ratings)

#print(data.userId.unique().shape[0]) #138493

#print(data.title.unique().shape[0]) #26729

#print(data.movieId.unique().shape[0]) #26744



data.shape

data = data.iloc[:3000000,:]

#print(data.userId.unique().shape[0]) #137065

#print(data.movieId.unique().shape[0]) #886
pivot_table = pd.pivot_table(data, index = ["userId"],columns = ["title"],values = "rating", fill_value=0, dropna=True)

print(pivot_table)
def findksimilarusers(user_id, pivot_ratings, k=50):

    similarities=[]

    indices=[]

    model_knn = NearestNeighbors(metric = cosine, algorithm = 'brute') 

    model_knn.fit(pivot_ratings)



    distances, indices = model_knn.kneighbors(pivot_ratings.loc[user_id].values.reshape(1, -1), n_neighbors = k+1)

    print(distances)

    print(indices)

    print(indices.shape)

    print(indices.flatten().shape)

    similarities = 1-distances.flatten()

    print(similarities)

    print(indices.flatten())

    print(similarities.shape)

    print(len(indices.flatten()))

    print(indices.flatten()[1])

    print ('{0} most similar users for User {1}:\n'.format(k,user_id))

    for i in range(0, len(indices.flatten())):

        if indices.flatten()[i] == user_id:

            continue;



        else:

            print ('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i], similarities[i]))

            

    return similarities,indices



similarities,indices = findksimilarusers(4,pivot_table)
def recommendations(user_id, similarities, indices):

    actual_indices = [x for x in indices.flatten()]

#    print(actual_indices)

    sim_ind = dict(zip(actual_indices, similarities))

    #print(sim_ind)



    del sim_ind[user_id-1]

    print(sim_ind)

    max_ind = [key for m in [max(sim_ind.values())] for key,val in sim_ind.items() if val == m]

    #print(max_ind)

    

    for maxSim_row in max_ind:

        similarUser = pivot_table.loc[maxSim_row,:]

        currUser = pivot_table.loc[user_id,:]   

#        print(currUser)

#        print(similarUser)

        currUser_zeros = currUser[currUser==0]

        print(currUser_zeros)

        for k in currUser_zeros.keys():

            if similarUser[k] != 0:

                print('Recommendations for User {0}: {1}'.format(user_id, k))

recommendations(4, similarities, indices)
x = (pivot_table.loc[4]).values.reshape(1,-1)

y = (pivot_table.loc[38500]).values.reshape(1,-1)

print(x.shape)

print(y.shape)

print(sklearn.metrics.pairwise.cosine_similarity(x,y))