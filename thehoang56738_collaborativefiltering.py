import pandas as pd 

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from scipy import sparse 
# a = ['python','math','DSA','ML','DL']
class CF(object):

    """docstring for CF

    input: matrix ['user_id', 'item_id', 'score']

    k: số item được recommend"""

    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):

        self.uuCF = uuCF # user-user (1) or item-item (0) CF

        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]

        self.k = k

        self.dist_func = dist_func

        self.Ybar_data = None

        # number of users and items. Remember to add 1 since id starts from 0

        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 

        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    

    def add(self, new_data):

        """

        Update Y_data matrix when new ratings come.

        For simplicity, suppose that there is no new user or item.

        """

        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

    

    def normalize_Y(self):

        users = self.Y_data[:, 0] # all users - first col of the Y_data

        self.Ybar_data = self.Y_data.copy()

        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):

            # row indices of rating done by user n

            # since indices need to be integers, we need to convert

            ids = np.where(users == n)[0].astype(np.int32)

            # indices of all ratings associated with user n

            item_ids = self.Y_data[ids, 1] 

            # and the corresponding ratings 

            ratings = self.Y_data[ids, 2]

            # take mean

            m = np.mean(ratings) 

            if np.isnan(m):

                m = 0 # to avoid empty array and nan value

            self.mu[n] = m

            # normalize

            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ################################################

        # form the rating matrix as a sparse matrix. Sparsity is important 

        # for both memory and computing efficiency. For example, if #user = 1M, 

        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 

        # you may not have enough memory to store this. Then, instead, we store 

        # nonzeros only, and, of course, their locations.

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],

            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))

#         print(self.Ybar.tocsr())

        self.Ybar = self.Ybar.tocsr()



    def similarity(self):

        eps = 1e-6

        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

        print("Similarity matrix\n",self.S)

    

        

    def refresh(self):

        """

        Normalize data and calculate similarity matrix again (after

        some few ratings added)

        """

        self.normalize_Y()

        self.similarity() 

        

    def fit(self):

        self.refresh()

        

    

    def __pred(self, u, i, normalized = 1):

        """ 

        predict the rating of user u for item i (normalized)

        if you need the un

        """

        # Step 1: find all users who rated i

        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)

        # Step 2: 

        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # Step 3: find similarity btw the current user and others 

        # who already rated i

        sim = self.S[u, users_rated_i]

        # Step 4: find the k most similarity users

        a = np.argsort(sim)[-self.k:] 

        # and the corresponding similarity levels

        nearest_s = sim[a]

        # How did each of 'near' users rated item i

        r = self.Ybar[i, users_rated_i[a]]

        if normalized:

            # add a small number, for instance, 1e-8, to avoid dividing by 0

            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)



        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    

    def pred(self, u, i, normalized = 1):

        """ 

        predict the rating of user u for item i (normalized)

        if you need the un

        """

        if self.uuCF: return self.__pred(u, i, normalized)

        return self.__pred(i, u, normalized)

            

    

    def recommend(self, u):

        """

        Determine all items should be recommended for user u.

        The decision is made based on all i such that:

        self.pred(u, i) > 0. Suppose we are considering items which 

        have not been rated by u yet. 

        """

        ids = np.where(self.Y_data[:, 0] == u)[0]

        items_rated_by_u = self.Y_data[ids, 1].tolist()              

        recommended_items = []

        for i in range(self.n_items):

            if i not in items_rated_by_u:

                rating = self.__pred(u, i)

                if rating > 0: 

                    recommended_items.append(i)

        

        return recommended_items 



    def print_recommendation(self):

        """

        print all items which should be recommended for each user 

        """

        print('Recommendation: ')

        for u in range(self.n_users):

            recommended_items = self.recommend(u)

            if self.uuCF:

#                 temp = [a[i] for i in recommended_items]

                print(' Recommend item(s):', recommended_items , 'for user', u)

            else: 

                print(' Recommend item', u, 'for user(s) : ', recommended_items)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']



ratings_base = pd.read_csv('../input/movielens-100k-dataset/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')

ratings_test = pd.read_csv('../input/movielens-100k-dataset/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')



rate_train = ratings_base.as_matrix()

rate_test = ratings_test.as_matrix()



# indices start from 0

rate_train[:, :2] -= 1

rate_test[:, :2] -= 1
rs = CF(rate_train, k = 30, uuCF = 1)

rs.fit()



n_tests = rate_test.shape[0]

SE = 0 # squared error

for n in range(n_tests):

    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)

    SE += (pred - rate_test[n, 2])**2 



RMSE = np.sqrt(SE/n_tests)

print('User-user CF, RMSE =', RMSE)
import pandas as pd 

import numpy as np

from numpy import linalg as LA

from sklearn.metrics.pairwise import cosine_similarity

from scipy import sparse 



class SVD_RS(object):

    """docstring for CF

    input: matrix['user_id', 'item_id', 'score']

    k: số lượng item được recommend """

    def __init__(self, Y_data, K, user_based = 1):

        self.Y_data = Y_data

        self.K = K

        self.user_based = user_based

        # number of users and items. Remember to add 1 since id starts from 0

        self.n_users = int(np.max(Y_data[:, 0])) + 1 

        self.n_items = int(np.max(Y_data[:, 1])) + 1            

        #self.all_users = self.Y_data[:,0] # all users (may be duplicated)

        self.n_ratings = Y_data.shape[0]

        # normalized data

        self.Ybar_data = self.Y_data.copy().astype(np.float32)



    def normalize_Y(self):

        if self.user_based:

            user_col = 0

            item_col = 1

            n_objects = self.n_users



        # if we want to normalize based on item, just switch first two columns of data

        else: # item bas

            user_col = 1

            item_col = 0 

            n_objects = self.n_items



        users = self.Y_data[:, user_col] 

        self.mu = np.zeros((n_objects,))

        for n in range(n_objects):

            # row indices of rating done by user n

            # since indices need to be integers, we need to convert

            ids = np.where(users == n)[0].astype(np.int32)

            # indices of all ratings associated with user n

            item_ids = self.Y_data[ids, 1] 

            # and the corresponding ratings 

            ratings = self.Y_data[ids, 2].astype(np.float32)

            # take mean

            m = np.mean(ratings) 

            if np.isnan(m):

                m = 0 # to avoid empty array and nan value

            self.mu[n] = m

            # normalize

            self.Ybar_data[ids, 2] = ratings - self.mu[n]

#             print self.Ybar_data



        ################################################

        # form the rating matrix as a sparse matrix. Sparsity is important 

        # for both memory and computing efficiency. For example, if #user = 1M, 

        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 

        # you may not have enough memory to store this. Then, instead, we store 

        # nonzeros only, and, of course, their locations.

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],

            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))

        self.Ybar = self.Ybar.todense()





    

    def fit(self): 

        """

        matrix factorization using SVD

        """

        self.normalize_Y()

        U, S, V = LA.svd(self.Ybar)

        print(U.shape,S.shape,V.shape)

        Uk = U[:, :self.K]

        Sk = S[:self.K]

        Vkt = V[:self.K, :]

        self.res = Uk.dot(np.diag(Sk)).dot(Vkt)

        

    def pred(self, u, i):

        """ 

        predict the rating of user u for item i 

        if you need the un

        """

        u = int(u)

        i = int(i)

        

        if self.user_based:

            bias = self.mu[u]

        else: 

            bias = self.mu[i]

        pred = self.res[i, u] + bias 

        if pred < 1:

            return 1 

        if pred > 5: 

            return 5 

        return pred 

        

    

    def pred_for_user(self, user_id):

        ids = np.where(self.Ybar_data[:, 0] == user_id)[0]

        items_rated_by_u = self.Ybar_data[ids, 1].tolist()              

        

        y_pred = self.res[:,user_id] + self.mu[user_id]

        predicted_ratings= []

        for i in range(self.n_items):

            if i not in items_rated_by_u:

                predicted_ratings.append((y_pred[i],i))

        return list(reversed(predicted_ratings))

    

    def evaluate_RMSE(self, rate_test):

        n_tests = rate_test.shape[0]

        SE = 0 # squared error

        for n in range(n_tests):

            pred = self.pred(rate_test[n, 0], rate_test[n, 1])

            SE += (pred - rate_test[n, 2])**2 



        RMSE = np.sqrt(SE/n_tests)

        return RMSE

    def print_recommendation(self):

        """

        print all items which should be recommended for each user 

        """

        print('Recommendation: ')

        for u in range(self.n_users):

            recommended_items = self.pred_for_user(u)

            if self.user_based:

                temp = [a[i[1]] for i in recommended_items[:self.K] ]

                print(' Recommend item(s):', temp , 'for user', u)

            else: 

                print(' Recommend item', a[u], 'for user(s) : ', recommended_items)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']



ratings_base = pd.read_csv('../input/movielens-100k-dataset/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')

ratings_test = pd.read_csv('../input/movielens-100k-dataset/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')



rate_train = ratings_base.as_matrix()

rate_test = ratings_test.as_matrix()



# indices start from 0

rate_train[:, :2] -= 1

rate_test[:, :2] -= 1
rs = SVD_RS(rate_train, K = 10, user_based = 1)

rs.fit()

# evaluate on test data

RMSE = rs.evaluate_RMSE(rate_test)

print('\nUser-based MF, RMSE =', RMSE)