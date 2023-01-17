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
        self.b = np.random.randn(data.n_items)
        self.U = np.random.randn(K, data.n_users)
        self.d = np.random.rand(data.n_users)    
        
    def updateU(self):
        for n in range(self.data.n_users):
        # Matrix items include all items is rated by user n
            i_rated = np.where(self.data.Ystad[:, n] != 0)[0] #item's index rated by n
            In = self.I[i_rated, :]
            b_n = self.b[i_rated]
            if In.shape[0] == 0:
                self.U[:, n] = 0
            else: 
                s = In.shape[0]
                u_n = self.U[:, n]
                y_n = self.data.Ystad[i_rated, n]
                #update u_n
                grad = -1/s * np.dot(In.T,(y_n - np.dot(In, u_n) - b_n - self.d[n]*np.ones_like(i_rated))) + self.lamda*u_n
                self.U[:, n] -= self.theta*grad
                #update b_n
                grad_bn = -1/s * (y_n - np.dot(In, u_n) - b_n - self.d[n]*np.ones_like(i_rated)) + self.lamda*b_n
                self.b[i_rated] -= self.theta*grad_bn
         
    def updateI(self):
        for m in range(self.data.n_items):
        # Matrix users who rated into item m
            i_rated = np.where(self.data.Ystad[m, :] != 0)[0] #user's index rated into m
            Um = self.U[:, i_rated]
            d_m = self.d[i_rated]
            if Um.shape[1] == 0: 
                self.I[m, :] = 0
            else:
                s = Um.shape[1]
                i_m = self.I[m, :]
                y_m = self.data.Ystad[m, i_rated]
                #update i_m
                grad = -1/s * np.dot(y_m - np.dot(i_m, Um)-self.b[m]*np.ones_like(i_rated)-d_m, Um.T) + self.lamda*i_m
                self.I[m, :] -= self.theta*grad
                #update d_m
                grad_dm = -1/s * (y_m - np.dot(i_m, Um)-self.b[m]*np.ones_like(i_rated)-d_m) + self.lamda*d_m
                self.d[i_rated] -= self.theta*grad_dm
    
    def sum_matrix(self, b, d):
        return np.dot(b.reshape(-1, 1), np.ones((1, d.shape[0]))) + np.dot(np.ones((b.shape[0], 1)), d.reshape(1, -1))

    
    def pred(self, I, U):
        #predict utility matrix base on formula Yhat = I.U
        Yhat = np.dot(I, U) + self.sum_matrix(self.b, self.d)
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
        loss_value = 1/(2*n_ratings)*np.linalg.norm(error, 'fro')**2 + \
        self.lamda/2*(np.linalg.norm(self.I, 'fro')**2 + \
                      np.linalg.norm(self.U, 'fro')**2 + \
                      np.sum(self.b**2) + np.sum(self.d**2))
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
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import math as math
columns = ['user_id', 'item_id', 'rating', 'timestamp']
movie_length = pd.read_csv('../input/ratings.dat', header = 0, \
                           names = columns, sep = '::', engine = 'python')
movie_length = movie_length.sort_values(['user_id', 'item_id'])
data = Data(dataset = movie_length, split_rate = 2/3)
model = Model(data = data, theta = 0.5, lamda = 0.1, K = 4)
mf = MF(data = data, model = model, n_iter = 1000, print_log_iter = 10)
mf.fit()