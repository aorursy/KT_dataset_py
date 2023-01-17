import numpy as np # linear algebra

import pandas as pd # data processing

import math
#data is the movie rating by users

#Rows define movies

#Columns define users

#row,column pair defines ratings given by user

#PROBLEM IS TO PREDICT RATINGS FOR MOVIES WHICH A USER HAS NOT GIVEN ANY RATINGS

data = pd.read_csv('../input/Movie.csv')
data
data.head()
no_movies = len(data['Ayush'])
no_users = len(data.columns)

no_users
#X for every movie has 3 Parameters: 1, Romance, Action

#Initializing X with small random values

X = np.random.uniform(low=0.01, high=0.25, size=(3,no_movies))

X[0,:] = 1

X
#Every User has similarly 3 coefficients

#Initializing theta with small random values

theta = np.random.uniform(low=0.01, high=0.25, size=(3,no_users))
#sumx to temporarily store values

sumx = np.zeros((3,1))

sumx
#Gradient Descent without regularization

for repeat in range(500):

    for j in range(no_users):

        for i in range(no_movies):

            sumx = np.zeros((3,1))

            for k in range(no_users):

                if(math.isnan(data.iloc[i,k])==False):

                    sumx[:,0] = sumx[:,0] + (np.dot(theta[:,k],X[:,i])-data.iloc[i,k])*theta[:,k]

                

            X[:,i:i+1]=X[:,i:i+1]-0.05*(sumx)

            sumx = np.zeros((3,1))

            for l in range(no_movies):

                if(math.isnan(data.iloc[l,j])==False):

                    sumx[:,0] = sumx[:,0] + (np.dot(theta[:,j],X[:,l])-data.iloc[l,j])*X[:,l]

        theta[:,j:j+1]=theta[:,j:j+1]-0.05*(sumx)

        sumx = np.zeros((3,1))

#After Gradient Descent X Value

X
#After Gradient Descent theta Value

theta
#Predicted User Ratings

#Note: -0 Means very small negative value

for j in range(no_movies):

    for i in range(no_users):

        predict = np.dot(theta[:,i],X[:,j])

        print(" %6.1f " % predict,end='  ')

    print()