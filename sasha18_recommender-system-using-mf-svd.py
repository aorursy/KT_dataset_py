from collections import defaultdict #acts just like dictionary except for the fact that it never raises a KeyError. It displays a default value for the Key that doesn't exist. Useful as many movies wouldn't have ratings as users did not watch them so this would avoid KeyErrors.
from surprise import SVD, Dataset #surprise is a scikit library used for recommendation systems
import pandas as pd

from surprise.prediction_algorithms.matrix_factorization import SVD
from sklearn.model_selection import train_test_split
import surprise
# Load movielens dataset.
data = Dataset.load_builtin('ml-100k')
data
#We will use entire data into training set
trainset = data.build_full_trainset()
trainset.ur #ur - user ratings
#Import packages for SVD
from numpy import array, diag, dot
from scipy.linalg import svd

#Define a sample matrix to be decomposed by SVD
A = array([[1,2,3],[4,5,6],[7,8,9]])
A
#Applying SVD on A would output 3 parameters namely
U,s,VT = svd(A)
print("U = ",U) #Orthogonal matrix
print('************************************************')
print("S = ",s) #Singular values
print('************************************************')
print("VT = ", VT) #Transpose of Orthogonal matrix

#Converting all singular values into diagonal matrix
sigma = diag(s)
print('Sigma = ',sigma)
B = U.dot(sigma.dot(VT))
print(B)
#Initialize and Fit SVD into trainset
algo = SVD()
algo.fit(trainset)
# Create testset - all movies not available in trainset
testset = trainset.build_anti_testset()
testset
#Predict ratings for movies in testset
pred = algo.test(testset)
pred
def get_top_n_movies(pred, n):
    #Write a function that map predictions to each user
    top_n = defaultdict(list) #convert list into defaultdict which accomadates empty key values pair
    for uid, iid, true_r,est, _ in pred:#variable names to all features in predictions
        top_n[uid].append((iid,est))
        
    #Sort the predictions and retrieve n highest scores
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key = lambda x:x[1], reverse = True)
        top_n[uid] = user_ratings[:n]
        
    return top_n

n = 10
top_n = get_top_n_movies(pred,n)
top_n
#Finally recommend a list of movies to user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])