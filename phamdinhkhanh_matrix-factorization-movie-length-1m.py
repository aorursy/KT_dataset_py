import pandas as pd
import numpy as np
columns = ['user_id', 'item_id', 'rating', 'timestamp']
movie_length = pd.read_csv('../input/ratings.dat', header = 0, \
                           names = columns, sep = '::', engine = 'python')
movie_length = movie_length.sort_values(['user_id', 'item_id'])
movie_length.head()
print('Data movie length shape: %s'%str(movie_length.shape))
print('No customers: %s'%str(np.unique(movie_length.iloc[:, 0]).shape[0]))
print('No movies: %s'%str(np.unique(movie_length.iloc[:, 1]).shape[0]))
movie_length['user_id'].value_counts().describe()
import matplotlib.pyplot as plt
%matplotlib inline
movie_length[['user_id', 'item_id']].groupby(['user_id']).count().\
hist(bins = 20, figsize = (12, 8))
plt.title('Distribution of no ratings by each customer')
plt.xlabel('No ratings')
plt.ylabel('No customers')
movie_length['item_id'].value_counts().describe()
movie_length[['user_id', 'item_id']].groupby(['item_id']).count().\
hist(bins = 20, figsize = (12, 8))
plt.title('Distribution of no ratings per each movie')
plt.xlabel('No ratings')
plt.ylabel('No movies')
#declare split_rate for train/total ratings
split_rate = 2/3

def split_train_test(dataset):
    gb = dataset.groupby('user_id')
    ls = [gb.get_group(x) for x in gb.groups]
    items = [x for x in gb.groups]
    index_size = [{'i': i, 'index':gb.groups[i], 'size':len(gb.groups[i])} for i in items]
    index_train = pd.Int64Index([])
    index_test = pd.Int64Index([])
    for x in index_size:
        np.random.shuffle(x['index'].values)
        le = int(x['size']*split_rate)
        index_train = index_train.append(x['index'][:le])
        index_test = index_test.append(x['index'][le:])
    train = dataset.iloc[index_train].values
    test = dataset.iloc[index_test].values
    #minus id to 1 to index start from 0
    train[:, 0] -= 1
    train[:, 1] -= 1
    test[:, 0] -= 1
    test[:, 1] -= 1
    return train, test

train, test = split_train_test(movie_length)
n_users = np.max(train[:, 0] + 1) #plus one because index start from 0
n_items = np.max(train[:, 1] + 1)
n_ratings = train.shape[0]
print('N user dimesion: %s'%n_users)
print('M item dimesion: %s'%n_items)
print('S Number of rating: %s'%n_ratings)
K = 2
theta = 0.75
lamda = 0.2
#Inititalize random matrix according to Gauss distribution
I = np.random.randn(n_items, K)
U = np.random.randn(K, n_users)
import scipy.sparse as sparse
#Rating matrix
Y = np.zeros(shape = (n_items, n_users))
print('Y utility matrix shape: %s'%str(Y.shape))
Y = sparse.coo_matrix((train[:, 2], (train[:, 1], train[:, 0])),\
                      shape = (n_items, n_users), dtype = np.float).toarray()
R = sparse.coo_matrix((np.ones((n_ratings,)), (train[:, 1], train[:, 0])),\
                      shape = (n_items, n_users)).toarray()
def standardize_Y(Y):
    sum_rating = Y.sum(axis = 0)
    u_rating = np.count_nonzero(Y, axis = 0)
    u_mean = sum_rating/u_rating
    for n in range(n_users):
        for m in range(n_items):
            if Y[m, n] != 0:
                Y[m, n] -= u_mean[n]
    return Y, u_mean

Y_stad, u_mean = standardize_Y(Y)
def updateU(U):
    for n in range(n_users):
    # Matrix items include all items is rated by user n
        i_rated = np.where(Y_stad[:, n] != 0)[0] #item's index rated by n
        In = I[i_rated, :]
        if In.shape[0] == 0:
            U[:, n] = 0
        else: 
            s = In.shape[0]
            u_n = U[:, n]
            y_n = Y_stad[i_rated, n]
            grad = -1/s * np.dot(In.T,(y_n-np.dot(In, u_n))) + lamda*u_n
            U[:, n] -= theta*grad
    return U
def updateI(I):
    for m in range(n_items):
    # Matrix users who rated into item m
        i_rated = np.where(Y_stad[m, :] != 0)[0] #user's index rated into m
        Um = U[:, i_rated]
        if Um.shape[1] == 0: 
            I[m, :] = 0
        else:
            s = Um.shape[1]
            i_m = I[m, :]
            y_m = Y_stad[m, i_rated]
            grad = -1/s * np.dot(y_m - np.dot(i_m, Um), Um.T) + lamda*i_m
            I[m, :] -= theta*grad
    return I
def pred(U, I):
    #predict utility matrix base on formula Y_hat = I.U
    Y_hat = np.dot(I, U)
    #invert to forecast values by plus user's mean ratings
    for n in range(n_users):
        Y_hat[:, n] += u_mean[n]
    #convert to interger values because of rating is integer
    Y_hat = Y_hat.astype(np.int32) 
    #replace values > 5 by 5 and values < 1 by 1
    Y_hat[Y_hat > 5] = 5
    Y_hat[Y_hat < 1] = 1
    return Y_hat

def pred_train_test(Y_hat, R):
    #replace values have not yet rated by 0 
    Y_pred = Y_hat.copy()
    Y_pred[R == 0] = 0
    return Y_pred
def loss(Y, Y_hat):
    error = Y-Y_hat
    loss_value = 1/(2*n_ratings)*np.linalg.norm(error, 'fro')**2 + \
    lamda/2*(np.linalg.norm(I, 'fro')**2 + np.linalg.norm(U, 'fro')**2)
    return loss_value
Y_test = sparse.coo_matrix((test[:, 2], (test[:, 1], test[:, 0])), \
                           shape = (n_items, n_users), dtype = np.float).toarray()
R_test = sparse.coo_matrix((np.ones(test.shape[0]), (test[:, 1], test[:, 0])), \
                           shape = (n_items, n_users), dtype = np.float).toarray()
import math
def RMSE(Y_test, Y_pred):
    error = Y_test - Y_pred
    n_ratings = test.shape[0]
    rmse = math.sqrt(np.linalg.norm(error, 'fro')**2/n_ratings)
    return rmse
def fit(Umatrix, Imatrix, Ytrain, Ytest, n_iter, log_iter):
    for i in range(n_iter):
        #update U and I
        Umatrix = updateU(Umatrix)
        Imatrix = updateI(Imatrix)
        #calculate Y_hat
        Y_hat = pred(Umatrix, Imatrix)
        #calculate Y_hat_train by replace non ratings by 0
        Y_pred_train = pred_train_test(Y_hat, R)
        #calculate loss function
        loss_value = loss(Ytrain, Y_pred_train)
        #calculate Y_pred on test dataset
        Y_pred_test = pred_train_test(Y_hat, R_test)
        #calculate RMSE
        rmse = RMSE(Ytest, Y_pred_test)
        if i % log_iter == 0:
            print('Iteration: {}; RMSE: {}; Loss value: {}'.format(i, rmse, loss_value))
    return Y_hat, Y_pred_test   
# Y_hat, Y_pred = fit(Umatrix = U, Imatrix = I, Ytrain = Y, Ytest = Y_test, n_iter = 100, log_iter = 10)
class Data(object):
    """
    This class used to manage data.
    Two arguments:
    dataset: pandas data frame include user_id, item_id and rating
    split_rate: number train ratings/ total ratings
    """
    def __init__(self, dataset, split_rate):
        self.dataset = dataset
        self.split_rate = split_rate
        self.train, self.test = self.split_train_test(self.dataset)
        self.n_users = np.max(self.train[:, 0] + 1) #plus one because index start from 0
        self.n_items = np.max(self.train[:, 1] + 1)
        self.Ytrain, self.Rtrain = self.utility_matrix(self.train)
        self.Ytest , self.Rtest  = self.utility_matrix(self.test)
        self.Ystad,  self.u_mean = self.standardize_Y(self.Ytrain)
        self.n_ratings = self.train.shape[0]
        
    def split_train_test(self, dataset):
        "split train and test"
        gb = dataset.groupby('user_id')
        ls = [gb.get_group(x) for x in gb.groups]
        items = [x for x in gb.groups]
        index_size = [{'i': i, 'index':gb.groups[i], 'size':len(gb.groups[i])} for i in items]
        index_train = pd.Int64Index([])
        index_test = pd.Int64Index([])
        for x in index_size:
            np.random.shuffle(x['index'].values)
            le = int(x['size']*self.split_rate)
            index_train = index_train.append(x['index'][:le])
            index_test = index_test.append(x['index'][le:])
        train = dataset.iloc[index_train].values
        test = dataset.iloc[index_test].values
        #minus id to 1 to index start from 0
        train[:, 0] -= 1
        train[:, 1] -= 1
        test[:, 0] -= 1
        test[:, 1] -= 1
        return train, test
    
    def utility_matrix(self, data_mtx):
        "create Y and R matrix"
        Y = np.zeros(shape = (self.n_items, self.n_users))
        Y = sparse.coo_matrix((data_mtx[:, 2], (data_mtx[:, 1], data_mtx[:, 0])), \
                              shape = (self.n_items, self.n_users), dtype = np.float).toarray()
        R = sparse.coo_matrix((np.ones((data_mtx.shape[0],)), (data_mtx[:, 1], data_mtx[:, 0])), \
                              shape = (self.n_items, self.n_users)).toarray()
        return Y, R
    
    def standardize_Y(self, Y):
        "standard data to mean ratings of each user = 0"
        sum_rating = Y.sum(axis = 0)
        u_rating = np.count_nonzero(Y, axis = 0)
        u_mean = sum_rating/u_rating
        for n in range(self.n_users):
            for m in range(self.n_items):
                if Y[m, n] != 0:
                    Y[m, n] -= u_mean[n]
        return Y, u_mean
class Model():
    """
    This class manage update U and I matrix, predict and evaluate error
    Four arguments:
    data: instance from Data class which supplies the data for model
    theta: learning rate
    lamda: regularization parameter
    K: number of latent factors
    """
    def __init__(self, data, theta, lamda, K):
        self.data = data
        self.theta = theta
        self.lamda = lamda
        self.K = K
        self.I = np.random.randn(data.n_items, K)
        self.U = np.random.randn(K, data.n_users)
        
               
    def updateU(self):
        for n in range(self.data.n_users):
        # Matrix items include all items is rated by user n
            i_rated = np.where(self.data.Ystad[:, n] != 0)[0] #item's index rated by n
            In = self.I[i_rated, :]
            if In.shape[0] == 0:
                self.U[:, n] = 0
            else: 
                s = In.shape[0]
                u_n = self.U[:, n]
                y_n = self.data.Ystad[i_rated, n]
                grad = -1/s * np.dot(In.T,(y_n-np.dot(In, u_n))) + self.lamda*u_n
                self.U[:, n] -= self.theta*grad
         
    def updateI(self):
        for m in range(self.data.n_items):
        # Matrix users who rated into item m
            i_rated = np.where(self.data.Ystad[m, :] != 0)[0] #user's index rated into m
            Um = self.U[:, i_rated]
            if Um.shape[1] == 0: 
                self.I[m, :] = 0
            else:
                s = Um.shape[1]
                i_m = self.I[m, :]
                y_m = self.data.Ystad[m, i_rated]
                grad = -1/s * np.dot(y_m - np.dot(i_m, Um), Um.T) + self.lamda*i_m
                self.I[m, :] -= self.theta*grad
    
    def pred(self, I, U):
        #predict utility matrix base on formula Yhat = I.U
        Yhat = np.dot(I, U)
        #invert to forecast values by plus user's mean ratings
        for n in range(self.data.n_users):
            Yhat[:, n] += self.data.u_mean[n]
        #convert to interger values because of rating is integer
        Yhat = Yhat.astype(np.int32) 
        #replace values > 5 by 5 and values < 1 by 1
        Yhat[Yhat > 5] = 5
        Yhat[Yhat < 1] = 1
        return Yhat

    def pred_train_test(self, Yhat, R):
        #replace values have not yet rated by 0 
        Y_pred = Yhat.copy()
        Y_pred[R == 0] = 0
        return Y_pred
    
    def loss(self, Y, Yhat):
        error = Y-Yhat
        n_ratings = np.sum(Y != 0)
        loss_value = 1/(2*n_ratings)*np.linalg.norm(error, 'fro')**2 +\
        self.lamda/2*(np.linalg.norm(self.I, 'fro')**2 + \
                 np.linalg.norm(self.U, 'fro')**2)
        return loss_value
    
    def RMSE(self, Y, Yhat):
        error = Y - Yhat
        n_ratings = np.sum(Y != 0)
        rmse = math.sqrt(np.linalg.norm(error, 'fro')**2/n_ratings)
        return rmse
class MF():
    """
    This class used to manage model and data
    Two main arguments:
    data: control the data
    model: control the functions which execute model
    """
    def __init__(self, data, model, n_iter, print_log_iter):
        self.data = data
        self.model = model
        self.n_iter = n_iter
        self.print_log_iter = print_log_iter
        self.Y_pred_train = None
        self.Y_pred_test = None
        self.Yhat = None
        
    def fit(self):
        for i in range(self.n_iter):
            #update U and I
            self.model.updateU()
            self.model.updateI()
            #calculate Y_hat
            self.Yhat = self.model.pred(self.model.I, self.model.U)
            #calculate Y_pred_train by replace non ratings by 0
            self.Y_pred_train = self.model.pred_train_test(self.Yhat, self.data.Rtrain)
            self.Y_pred_test  = self.model.pred_train_test(self.Yhat, self.data.Rtest)
            if i % self.print_log_iter == 0:
                print('Iteration: {}; RMSE: {}; Loss value: {}'.\
                      format(i, self.model.RMSE(self.data.Ytest, self.Y_pred_test),\
                             self.model.loss(self.data.Ytrain, self.Y_pred_train)))
                
    def recommend_for_user(self, user_id, k_neighbors):
        recm = np.concatenate((np.arange(1, self.Y_pred_test.shape[0]+1).reshape(-1, 1), \
                               self.Y_pred_test[:, user_id - 1].reshape(-1, 1)), axis = 1)
        recm.sort(axis = 0)
        print('Top %s item_id recommended to user_id %s: %s'%\
              (k_neighbors, user_id, str(recm[-k_neighbors:, 0])))
data = Data(dataset = movie_length, split_rate = 2/3)
model = Model(data = data, theta = 0.75, lamda = 0.1, K = 3)
mf = MF(data = data, model = model, n_iter = 100, print_log_iter = 10)
mf.fit()
mf.recommend_for_user(user_id = 200, k_neighbors = 10)