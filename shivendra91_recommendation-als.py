# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.sparse as sp

from scipy.sparse import vstack

from scipy import sparse

from scipy.sparse.linalg import spsolve



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn.model_selection import train_test_split

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
events = pd.read_csv('../input/events.csv')

category_tree = pd.read_csv('../input/category_tree.csv')

items1 = pd.read_csv('../input/item_properties_part1.csv')

items2 = pd.read_csv('../input/item_properties_part2.csv')

items = pd.concat([items1, items2])
#user_with_buy = dict()

#index = 0

#for row in events.itertuples():

#    if row.event == 'addtocart' or row.event == 'transaction':

#        if row[2] not in user_with_buy:

#            user_with_buy[row[2]] = index

#            index = index + 1
n_users = events['visitorid'].unique().shape[0]
n_items = items['itemid'].max()
print (str(n_users) +" " +  str(n_items))
user_to_item_matrix = sp.dok_matrix((n_users+1, n_items+2), dtype=np.int8)
# We need to check whether we need to add the frequency of view, addtocart and transation.

# Currently we are only taking a single value for each row and column.



action_weights = [1,2,3]

for row in events.itertuples():

#    if row[2] not in user_with_buy:

#        continue

#    mapped_user_key = user_with_buy[row[2]]

    mapped_user_key = row[2]

    if row.event == 'view':

            user_to_item_matrix[mapped_user_key, row[4]] = action_weights[0]

    elif row.event == 'addtocart':

            user_to_item_matrix[mapped_user_key, row[4]] = action_weights[1]        

    elif row.event == 'transaction':

            user_to_item_matrix[mapped_user_key, row[4]] = action_weights[2]
user_to_item_matrix = user_to_item_matrix.tocsr()

print (user_to_item_matrix.shape)
sparsity = float(len(user_to_item_matrix.nonzero()[0]))

sparsity /= (user_to_item_matrix.shape[0] * user_to_item_matrix.shape[1])

sparsity *= 100

print (sparsity)
X_train, X_test = train_test_split(user_to_item_matrix, test_size=0.20)
X_train.shape
X_test.shape
from sklearn.metrics.pairwise import cosine_similarity
# TODO: this is user to user similarity. check item to item similarity as well

cosine_similarity_matrix = cosine_similarity(X_train, X_train, dense_output=False)

cosine_similarity_matrix.setdiag(0)
cosine_similarity_matrix_ll=cosine_similarity_matrix.tolil()
cosine_similarity_matrix.head()
def implicit_weighted_ALS(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):

    '''

    Implicit weighted ALS taken from Hu, Koren, and Volinsky 2008. Designed for alternating least squares and implicit

    feedback based collaborative filtering. 

    

    parameters:

    

    training_set - Our matrix of ratings with shape m x n, where m is the number of users and n is the number of items.

    Should be a sparse csr matrix to save space. 

    

    lambda_val - Used for regularization during alternating least squares. Increasing this value may increase bias

    but decrease variance. Default is 0.1. 

    

    alpha - The parameter associated with the confidence matrix discussed in the paper, where Cui = 1 + alpha*Rui. 

    The paper found a default of 40 most effective. Decreasing this will decrease the variability in confidence between

    various ratings.

    

    iterations - The number of times to alternate between both user feature vector and item feature vector in

    alternating least squares. More iterations will allow better convergence at the cost of increased computation. 

    The authors found 10 iterations was sufficient, but more may be required to converge. 

    

    rank_size - The number of latent features in the user/item feature vectors. The paper recommends varying this 

    between 20-200. Increasing the number of features may overfit but could reduce bias. 

    

    seed - Set the seed for reproducible results

    

    returns:

    

    The feature vectors for users and items. The dot product of these feature vectors should give you the expected 

    "rating" at each point in your original matrix. 

    '''

    

    # first set up our confidence matrix

    

    conf = (alpha*training_set) # To allow the matrix to stay sparse, I will add one later when each row is taken 

                                # and converted to dense. 

    num_user = conf.shape[0]

    num_item = conf.shape[1] # Get the size of our original ratings matrix, m x n

    

    # initialize our X/Y feature vectors randomly with a set seed

    rstate = np.random.RandomState(seed)

    

    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size))) # Random numbers in a m x rank shape

    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size))) # Normally this would be rank x n but we can 

                                                                 # transpose at the end. Makes calculation more simple.

    X_eye = sparse.eye(num_user)

    Y_eye = sparse.eye(num_item)

    lambda_eye = lambda_val * sparse.eye(rank_size) # Our regularization term lambda*I. 

    

    # We can compute this before iteration starts. 

    

    # Begin iterations

   

    for iter_step in range(iterations): # Iterate back and forth between solving X given fixed Y and vice versa

        # Compute yTy and xTx at beginning of each iteration to save computing time

        yTy = Y.T.dot(Y)

        xTx = X.T.dot(X)

        # Being iteration to solve for X based on fixed Y

        for u in range(num_user):

            conf_samp = conf[u,:].toarray() # Grab user row from confidence matrix and convert to dense

            pref = conf_samp.copy() 

            pref[pref != 0] = 1 # Create binarized preference vector 

            CuI = sparse.diags(conf_samp, [0]) # Get Cu - I term, don't need to subtract 1 since we never added it 

            yTCuIY = Y.T.dot(CuI).dot(Y) # This is the yT(Cu-I)Y term 

            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) # This is the yTCuPu term, where we add the eye back in

                                                      # Cu - I + I = Cu

            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 

            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper  

        # Begin iteration to solve for Y based on fixed X 

        for i in range(num_item):

            conf_samp = conf[:,i].T.toarray() # transpose to get it in row format and convert to dense

            pref = conf_samp.copy()

            pref[pref != 0] = 1 # Create binarized preference vector

            CiI = sparse.diags(conf_samp, [0]) # Get Ci - I term, don't need to subtract 1 since we never added it

            xTCiIX = X.T.dot(CiI).dot(X) # This is the xT(Cu-I)X term

            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) # This is the xTCiPi term

            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)

            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi, equation 5 from the paper

    # End iterations

    return X, Y.T # Transpose at the end to make up for not being transposed at the beginning. 

                         # Y needs to be rank x n. Keep these as separate matrices for scale reasons. 
! pip install implicit
user_vecs, item_vecs = implicit_weighted_ALS(X_train, lambda_val = 0.1, alpha = 15, iterations = 1,

                                            rank_size = 20)
def max_n(row_data, row_indices, n):

        i = row_data.argsort()[-n:]

        # i = row_data.argpartition(-n)[-n:]

        top_values = row_data[i]

        top_indices = row_indices[i]  # do the sparse indices matter?

        return top_values, top_indices, i
def predict_topk(ratings, similarity, kind='user', k=40):

    pred = sp.csr_matrix((0,ratings.shape[1]), dtype=np.int8)

    if kind == 'user':

        for i in range(similarity.shape[0]):

            top_k_values, top_k_users = max_n(np.array(similarity.data[i]),np.array(similarity.rows[i]),k)[:2]

            current = top_k_values.reshape(1,-1).dot(ratings[top_k_users].todense())

            current /= np.sum(np.abs(top_k_values))+1

            vstack([pred, current])

    if kind == 'item':

        for j in range(ratings.shape[1]):

            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]

            for i in range(ratings.shape[0]):

                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 

                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        

    

    return pred
pred = predict_topk(X_train, cosine_similarity_matrix_ll, kind='user', k=5)