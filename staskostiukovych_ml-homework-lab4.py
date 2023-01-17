import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import surprise  #Scikit-Learn library for recommender systems. 

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/steam-video-games/steam-200k.csv', header=None, index_col=False,names = ['UserID', 'Game', 'Action', 'Hours', 'Not Needed'])
df.head()
# Creating a new variable 'Hours Played' and code it as previously described.
df['Hours_Played'] = df['Hours'].astype('float32')

df.loc[(df['Action'] == 'purchase') & (df['Hours'] == 1.0), 'Hours_Played'] = 0

df.UserID = df['UserID'].astype('int')
df = df.sort_values(['UserID', 'Game', 'Hours_Played'])

clean_df = df.drop_duplicates(['UserID', 'Game'], keep = 'last').drop(['Action', 'Hours', 'Not Needed'], axis = 1)


data_matrix = clean_df.values


clean_df.head()
# # pd.qcut(clean_df['Hours_Played'], 5)
# pd.cut(clean_df['Hours_Played'].rank(method='first'), 5, labels=[1, 2 ,3, 4, 5]).value_counts()
n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())

print(f'Кількість унікальних юзерів:  {n_users}, ігор: {n_games}')
clean_df.Game.value_counts()
clean_df.UserID.value_counts()
sparsity = clean_df.shape[0] / float(n_users * n_games)
print('{:.2%} буде заповнено'.format(sparsity * 100))
user2idx = {user: i for i, user in enumerate(clean_df.UserID.unique())}
idx2user = {i: user for user, i in user2idx.items()}

game2idx = {game: i for i, game in enumerate(clean_df.Game.unique())}
idx2game = {i: game for game, i in game2idx.items()}
user_idx = clean_df['UserID'].apply(lambda x: user2idx[x]).values
game_idx = clean_df['Game'].apply(lambda x: game2idx[x]).values
hours = clean_df['Hours_Played'].values
user_idx.shape, game_idx.shape, hours.shape
data = np.hstack([user_idx.reshape(-1, 1), game_idx.reshape(-1, 1), hours.reshape(-1, 1)]).astype(np.int64)
data.shape
data
class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  

        pairs_train = train_vec.shape[0]  # traindata 
        
        pairs_test = test_vec.shape[0]  # testdata

        # 1-p-i, 2-m-c
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # user
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # item

        incremental = False  # 增量
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.normal(0,.1, (num_item, self.num_feat))  # numpy.random.randn M x D 
            self.w_User = 0.1 * np.random.normal(0,.1, (num_user, self.num_feat))  # numpy.random.randn N x D

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # M x D
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # N x D

        while self.epoch < self.maxepoch: 
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order) 

            # Batch update
            for batch in range(self.num_batches):
                

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int64')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int64')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]
                
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]
                    
# #                 test_123 = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size
#                 # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int64'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int64'), :]),
                                      axis=1)  # mean_inv subtracted
                    
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int64'), :],
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int64'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))

    def predict(self, invID):
        return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:] 

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int64'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)
train, test = train_test_split(data, test_size=0.2, random_state=4)
%%time
pmf10 = PMF()
pmf10.set_params({"num_feat": 10, "epsilon": 0.05, "_lambda": 0.00000001, "momentum": 0.05, "maxepoch": 10, "num_batches": 80,
                "batch_size": 5000})

pmf10.fit(train, test)
%%time
pmf30 = PMF()
pmf30.set_params({"num_feat": 30, "epsilon": 0.05, "_lambda": 0.000000001, "momentum": 0.05, "maxepoch": 10, "num_batches": 50,
                "batch_size": 3000})
pmf30.fit(train, test)
%%time
pmf50 = PMF()
pmf50.set_params({"num_feat": 50, "epsilon": 0.05, "_lambda": 0.0001, "momentum": 0.05, "maxepoch": 10, "num_batches": 30,
                "batch_size": 5000})
pmf50.fit(train, test)
%%time
pmf3 = PMF()
pmf3.set_params({"num_feat": 3, "epsilon": 0.05, "_lambda": 0.00000001, "momentum": 0.05, "maxepoch": 10, "num_batches": 50,
                "batch_size": 5000})

pmf3.fit(train, test)
pmf1 = PMF()
pmf1.set_params({"num_feat": 3, "epsilon": 0.05, "_lambda": 0.00000001, "momentum": 0.05, "maxepoch": 10, "num_batches": 50,
                "batch_size": 5000})

pmf1.fit(train, test)
precision, recall = pmf1.topK(test)
print(f'precision = {precision}, recall = {recall}')
class ProbabilisticMatrixFactorization(surprise.AlgoBase):
# Randomly initializes two Matrices, Stochastic Gradient Descent to be able to optimize the best factorization for ratings.
    def __init__(self,learning_rate,num_epochs,num_factors):
       # super(surprise.AlgoBase)
        self.alpha = learning_rate #learning rate for Stochastic Gradient Descent
        self.num_epochs = num_epochs
        self.num_factors = num_factors
    
    def fit(self,train):
        #randomly initialize user/item factors from a Gaussian
        P = np.random.normal(0,.1,(train.n_users,self.num_factors))
        Q = np.random.normal(0,.1,(train.n_items,self.num_factors))
        #print('fit')

        for epoch in range(self.num_epochs):
            for u,i,r_ui in train.all_ratings():
                residual = r_ui - np.dot(P[u],Q[i])
                temp = P[u,:] # we want to update them at the same time, so we make a temporary variable. 
                P[u,:] +=  self.alpha * residual * Q[i]
                Q[i,:] +=  self.alpha * residual * temp 

                
        self.P = P
        self.Q = Q

        self.trainset = train
    
    
    def estimate(self,u,i):
        #returns estimated rating for user u and item i. Prerequisite: Algorithm must be fit to training set.
        #check to see if u and i are in the train set:
        #print('gahh')

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            #print(u,i, '\n','yep:', self.P[u],self.Q[i])
            #return scalar product of P[u] and Q[i]
            nanCheck = np.dot(self.P[u],self.Q[i])
            
            if np.isnan(nanCheck):
                return self.trainset.global_mean
            else:
                return np.dot(self.P[u,:],self.Q[i,:])
        else:# if its not known we'll return the general average. 
           # print('global mean')
            return self.trainset.global_mean
data_surprise = surprise.Dataset.load_from_df(clean_df, surprise.Reader())
gs = surprise.model_selection.GridSearchCV(
    ProbabilisticMatrixFactorization, 
    param_grid={'learning_rate':[0.0001],
                'num_epochs':[50],
                'num_factors':[1, 2, 3]},
    measures=['rmse', 'mae'], 
    cv=4)

gs.fit(data_surprise)
gs.best_params
print('rsme: ',gs.best_score['rmse'],'mae: ',gs.best_score['mae'])
gs_upgrade = surprise.model_selection.GridSearchCV(
    ProbabilisticMatrixFactorization, 
    param_grid={'learning_rate':[0.0001, 0.01, 1e-6],
                'num_epochs':[50, 100],
                'num_factors':[1]},
    measures=['rmse', 'mae'], 
    cv=4)

gs_upgrade.fit(data_surprise)
print('rsme: ',gs_upgrade.best_score['rmse'],'mae: ',gs_upgrade.best_score['mae'])
print('rsme: ',gs_upgrade.best_params['rmse'],'mae: ',gs_upgrade.best_params['mae'])
best_params = gs_upgrade.best_params['rmse']
bestVersion = ProbabilisticMatrixFactorization(learning_rate=best_params['learning_rate'],num_epochs=best_params['num_epochs'],num_factors=best_params['num_factors'])
#we can use k-fold cross validation to evaluate the best model. 
kSplit = surprise.model_selection.KFold(n_splits=10,shuffle=True)
for train,test in kSplit.split(data_surprise):
    bestVersion.fit(train)
    prediction = bestVersion.test(test)
    surprise.accuracy.rmse(prediction,verbose=True)