import pandas as pd

import numpy as np
#Reading the file:

movie_dataset=pd.read_csv("../input/movie_ratings.csv")

#Dropping user name column:



movie_ratings=movie_dataset.drop('users', axis=1)

print(movie_ratings.head(5))

print(" Shape of movie matrix is :", movie_ratings.shape)
# Function for matrix factorization:



def matrix_factorization(R, P, Q, K, steps=10000, alpha=0.0002, beta=0.02):

    Q = Q.T

    for step in range(steps):

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):

                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])

                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

        #print(e)

        if e < 100:

            break

    return P, Q.T,e
#Defining all the parameters:



R = np.array(movie_ratings)

N = len(R)

M = len(R[0])

K = 3

P = np.random.rand(N,K)

#print(P.shape)

Q = np.random.rand(M,K) 

#print(Q.shape)



#Calling the function:



nP, nQ, e= matrix_factorization(R, P, Q, K)

print("Breakout Error form the function is", e)



#Combining it again as a full matrix

nR = np.dot(nP, nQ.T)

print(nR)
# Replacing all the movie rating by -1 that are already watched by users:



def eliminating_watched_movies(R):

    for i in range(len(R)):

            for j in range(len(R[i])):

                if not np.isnan(R[i][j]): 

                    R[i][j]=-1               

    return R
# Adding predictions to this matrix:



def Adding_predictions(R,nR):

    #First calling the function eliminating_watched_movies:

    eliminating_watched_movies(R)

    for i in range(len(R)):

            for j in range(len(R[i])):

                if np.isnan(R[i][j]): 

                    R[i][j]=nR[i][j]             

    return R
#Final Matrix of ratings to use for recommendation:



Adding_predictions(R,nR)
#Defing a list ranging 1-50 to add 50 users:



mylist = list(range(1,51))

print(mylist)



#Add indexes and column name:

Ratings=pd.DataFrame(R,columns=['movie1','movie2','movie3','movie4','movie5',

                                'movie6','movie7','movie8','movie9','movie10'],index=mylist)

print(Ratings)
#For each user recommendation ratings are sorted:



def sorted_ratings():

    for i in range(len(mylist)):

        a=Ratings.iloc[i]

        print(a.sort_values(ascending=False))

sorted_ratings()



#We can pick top 2 or top 3 movies to send recommendation to users.
def top_2_recommendations_for_user(user_id):

        #Using loc to get index based values:

        a=Ratings.loc[user_id].sort_values(ascending=False)

        if a[0]>0 and a[1]>0: 

            print(a[0:2])

        if a[0]>0 and a[1]<0: 

            print(a[0:1])

        if a[0]<0: 

            print("No Recommendation")

top_2_recommendations_for_user(3)   



#There is no user who has watched all the movies

#Try only one recommendation with movie 48

#Try two recommendation with movie 1