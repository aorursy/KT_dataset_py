import numpy as np

import pandas as pd

from sklearn.cluster import KMeans

from sklearn.cluster import OPTICS
anime_original = pd.read_csv('../input/anime-recommendations-database/anime.csv')

rating_original = pd.read_csv('../input/anime-recommendations-database/rating.csv')
anime = anime_original.copy()

rating = rating_original.copy()
condition = rating.loc[rating.rating == -1,:].index

rating.drop(condition,inplace=True)

rating
# Creating an array with the Id's of the rated animes

anime_id_list = np.unique(rating.anime_id.values)

anime_id_list
from scipy.sparse import csr_matrix

from pandas.api.types import CategoricalDtype



user_c = CategoricalDtype(sorted(rating.user_id.unique()), ordered=True)

anime_c = CategoricalDtype(sorted(rating.anime_id.unique()), ordered=True)



row = rating.user_id.astype(user_c).cat.codes

col = rating.anime_id.astype(anime_c).cat.codes

sparse_matrix = csr_matrix((rating["rating"], (row, col)), \

                           shape=(user_c.categories.size, anime_c.categories.size))
### Importing the sparse matrix previously calculated

import scipy.sparse

X_sparse = scipy.sparse.load_npz("../input/user-matrix/sparse_matrix.npz")
#Less than 1% of the entries belong to given ratings

sparcity = (X_sparse.nonzero()[0].shape[0])/(X_sparse.shape[0]*X_sparse.shape[1])

sparcity*100
def Gradient_Descent(X):

    X_sparse = X/np.max(X)           # normalizing values

    print(X_sparse)

    n_factors = 100

    n_steps =  500                # optimized

    alpha = 0.01                   # optimized

    

    #initializing the vectors randomly:

    p = np.random.normal(0, .01, (X_sparse.shape[0], n_factors))

    q = np.random.normal(0, .01, (n_factors, X_sparse.shape[1]))      # changed so as to follow matrix multiplication rule

    

    for k in range(0,n_steps):

        for (i,j) in zip(X_sparse.nonzero()[0],X_sparse.nonzero()[1]):

            err = X_sparse[i,j] - np.dot(p[i, :],q[:, j])              # multiply whole row and column

            p[i, :] = p[i, :] + alpha*q[:, j]*err                      # update whole row and column

            q[:, j] = q[:, j] + alpha*p[i, :]*err                      # update whole row and column

    print(np.dot(p, q)) 

    p = p*np.sqrt(np.max(X))                         # matrix multiplication rule for normalized values

    q = q*np.sqrt(np.max(X))                         # matrix multiplication rule for normalized values

    return (p,q)   
#p,q = Gradient_Descent(X_sparse)
#Loading both of the arrays

import numpy as np

p = np.load("../input/user-matrix/P_100.npy")

q = np.load("../input/user-matrix/Q_100.npy")

u = np.dot(p,q)

u
u.mean()
anime.loc[anime.name == 'Fullmetal Alchemist: Brotherhood']
np.where(anime_id_list == 5114 )
print('The mean is: ',u[:,3936].mean())

print('The standard deviation is: ',u[:,3936].std())
#function to reduce the ratings of the best rated anime

def squeezing_top_values(u):

  p1 = 9        # paremeter to reduce the greater than 9-mean animes

  p2 = 200      # paremeter to reduce the greater than 8 and lower than 9 mean animes

  index1 = np.where(u.mean(axis=0)>9)

  index2 = np.where((u.mean(axis=0)>8) & (u.mean(axis=0)<9))

  k = u.mean()

  for i in index1:

    u[:,i] = (u[:,i]*p1 + k)/(p1+1)

  for i in index2:

    u[:,i] = (u[:,i]*p2 + k)/(p2+1)

  return u





u = squeezing_top_values(u)    

# Now we have:

print('The mean now is: ',u[:,3936].mean())

print('The standard deviation now is: ',u[:,3936].std())



#That's better!
''' This function takes the names and the ratings of the animes given 

by the new user and transforms them to the correct form such that the algorithm

will understand

'''



def user_input_scores(array):

    a = []

    scores = []

    for i in range(0,len(array)):

        a.append(anime.loc[anime.name == array[i][0]].anime_id.values[0]) 

    a = np.array(a)

    

    for i in range(0,len(anime_id_list) - len(array)):

        scores.append(0)

    

    for i in range(0,len(a)):

        index = np.where(anime_id_list == a[i])[0][0]

        scores.insert(index,array[i][1])

    scores = np.array(scores)

    return scores

# shonen

y1 = [('Death Note',8),('Naruto',10),('Hunter x Hunter (2011)',9)]

# sports

y2 = [('Haikyuu!!',10),('Diamond no Ace',10)]



# Slice of Life

y3 = [('Toradora!',10),('Mob Psycho 100',10),('Suzumiya Haruhi no Yuuutsu',9)]



# Mecha

y4 = [('Code Geass: Hangyaku no Lelouch',10),('Neon Genesis Evangelion',10),('Guilty Crown',9)]



# Music

y5 = [('Shigatsu wa Kimi no Uso',9),('K-On!',10)]



#Kids

y6 = [('Pokemon',10),('Digimon Adventure',10)]
#Function to determine the positions of the animes that the user has seen:



def position_seen_animes(y):

  scores = user_input_scores(y)

  position = np.where(scores!=0)[0]

  return position



# Function to create an array from the users matrix with only the animes the new user has seen



def user_scores_seen_animes(X,y):

  positions = position_seen_animes(y)

  return X[:,positions]



#Function to apply Kmeans algorithm to get similar viewers



def similar_users_position(X,y):

  

  n_clusters = 8 ### The best number after exhaustive tests



  X_transformed = user_scores_seen_animes(X,y)

  y_transformed = user_input_scores(y)[position_seen_animes(y)].reshape(1,-1)

  kmeans = KMeans(n_clusters = n_clusters).fit(X_transformed)

  group = kmeans.predict(y_transformed)

    

  (array_labels,n_labels) = np.unique(kmeans.labels_,return_counts = True) #test to see if the clustering was well done

  #print(n_labels,group) -- test to see if the clustering was well done

  similar_user_position = np.where(kmeans.labels_ == group)[0]

  return similar_user_position





#Function to take the mean scores of the similar users in order to rank the animes



def mean_scores(X,y):

  similar_users_scores = []

  

  index = similar_users_position(X,y) #similar users positions on the user matrix

  scores = np.zeros(len(anime_id_list)) #inicializing 



  for i in index:

    similar_users_scores.append(X[i])



  similar_users_scores = np.array(similar_users_scores)



  mean_scores = similar_users_scores.sum(axis = 0)/similar_users_scores.shape[0] # Getting the mean_scores



  return mean_scores



#Function to determine the positions of the best rated shows



def best_rated_shows(X,y):

  mean_marks = mean_scores(X,y)

  index = np.flip(anime_id_list[np.argsort(mean_marks)[-10:]]) ### Choose how many recommendations will be displayed

  return index



#Function to determine the names of the best rated shows



def names_best_rated_shows(X,y):

  index = best_rated_shows(X,y)

  print('The best recommendations for you are:\n')

  for i in index:

        print(anime.loc[anime.anime_id == i,['name']].values[0][0])

        print('')













# Shonen

r1 = names_best_rated_shows(u,y1)
# Sport

r2 = names_best_rated_shows(u,y2)
# Slice of life

r3 = names_best_rated_shows(u,y3)
# Mecha

r4 = names_best_rated_shows(u,y4) 
# Music

r5 = names_best_rated_shows(u,y5)
# Kids

r6 = names_best_rated_shows(u,y6)