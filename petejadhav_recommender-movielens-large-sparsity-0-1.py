# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/ratings.csv')
df = df.drop(columns = ['timestamp'])
df.head(5)
n_users = df.userId.unique().shape[0]
n_items = df.movieId.unique().shape[0]
print(str(n_users) + ' users')
print(str(n_items) + ' movies')
ratings = np.zeros((df['userId'].max(), df['movieId'].max()))
ratings.shape
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
ratings
# Sparsity of rating matrix
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
sparsity = 100 - sparsity
print("Sparsity "+ str(sparsity) + " %")
def get_RMSE(Q, X, Y):
    return np.sqrt(np.sum((Q - np.dot(X, Y))**2))

def ALS_no_bias_no_reg(ratings, latent_factors = 8, iterations = 10):
    M,N = ratings.shape
    K = latent_factors
    U = np.random.randn(M, K)
    V = np.random.randn(K, N)
    errors = []
    for i in range(iterations):
        U = np.linalg.solve(np.dot(V, V.T), np.dot(V, ratings.T)).T
        V = np.linalg.solve(np.dot(U.T, U), np.dot(U.T, ratings))
        error = get_RMSE(ratings,U,V)
        print("Iteration - "+str(i)+" Error - "+str(error)) 
        errors.append(error)
    predictor = np.dot(U,V)
    return predictor, errors

def ALS_no_bias_with_reg(ratings, latent_factors = 8, reg_term = 0.1, iterations = 10):
    M,N = ratings.shape
    K = latent_factors
    U = np.random.randn(M, K)
    V = np.random.randn(K, N)
    errors = []
    for i in range(iterations):
        U = np.linalg.solve(np.dot(V, V.T) + reg_term * np.eye(K), np.dot(V, ratings.T)).T
        V = np.linalg.solve(np.dot(U.T, U) + reg_term * np.eye(K), np.dot(U.T, ratings))
        error = get_RMSE(ratings,U,V)
        print("Iteration - "+str(i)+" Error - "+str(error)) 
        errors.append(error)
    predictor = np.dot(U,V)
    return predictor, errors

def ALS_full(ratings, latent_factors = 8, reg_term = 0.1, iterations = 10):
    M,N = ratings.shape
    K = latent_factors
    U = np.random.randn(M, K)
    V = np.random.randn(K, N)
    B = np.random.randn(M,1)
    C = np.random.randn(1,N)
    errors = []
    for i in range(iterations):
        U = np.linalg.solve(np.dot(V, V.T) + reg_term * np.eye(K), np.dot(V , ratings.T - B.T - C.T)).T
        V = np.linalg.solve(np.dot(U.T, U) + reg_term * np.eye(K), np.dot(U.T, ratings - B - C))
        B = (ratings - np.dot(U,V) - C)/(1+reg_term)
        C = (ratings - np.dot(U,V) - B)/(1+reg_term)
        error = np.sqrt(np.sum((ratings - (np.dot(U, V) + B + C))**2))
        print("Iteration - "+str(i+1)+" Error - "+str(error)) 
        errors.append(error)
    predictor = np.dot(U,V) + B + C
    return predictor, errors
predictor_full,errors_full = ALS_full(ratings, latent_factors = 10, iterations = 20, reg_term = 0.1)
print(predictor_full[0,1424])
print(ratings[0,1424])
df.head(20)