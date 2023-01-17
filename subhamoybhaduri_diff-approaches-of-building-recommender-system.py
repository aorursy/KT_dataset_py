import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
#Reading Users file:
u_cols = ['User_ID', 'Age', 'Sex', 'Occupation', 'ZIP_Code']
users = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user',
                    sep='|', names=u_cols,encoding='latin-1')

#Reading Ratings file:
r_cols = ['User_ID', 'Movie_ID', 'Rating', 'Timestamp']
ratings = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data',
                      sep='\t', names=r_cols,encoding='latin-1')
users.shape
users.head()
ratings.shape
ratings.head()
len(ratings['Movie_ID'].unique())
nb_users  = users['User_ID'].nunique()
nb_movies = ratings['Movie_ID'].nunique()

print("There are %d unique users and %d unique movies; so we need to prepare " 
      "an matrix of size %d by %d." %(nb_users, nb_movies, nb_users, nb_movies))
ratings_matrix = ratings.pivot_table(index=['User_ID'],columns=['Movie_ID'],values='Rating').reset_index(drop=True)
ratings_matrix.fillna(0, inplace = True)

data_matrix = np.array(ratings_matrix)
print(data_matrix.shape)
print(data_matrix)
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import itertools
gmm_model = GaussianMixture(n_components=2, covariance_type='full', 
                            tol=0.001, reg_covar=1e-06, max_iter=100, 
                            n_init=1, init_params='kmeans', weights_init=None, 
                            means_init=None, precisions_init=None, random_state=42, 
                            warm_start=False, verbose=0, verbose_interval=10)
gmm_model.fit(data_matrix)
print(gmm_model.means_.shape)
print(gmm_model.covariances_.shape)
print(gmm_model.weights_.shape)
Image("../input/input/4.JPG")
Image("../input/input/5.JPG")
#Fill Missing Values i.e Recommend
inver0 = np.linalg.inv(gmm_model.covariances_[0])
inver1 = np.linalg.inv(gmm_model.covariances_[1])
deter0 = np.linalg.det(gmm_model.covariances_[0])
deter1 = np.linalg.det(gmm_model.covariances_[1])

n = data_matrix.shape[0]
d = data_matrix.shape[1]
K = gmm_model.means_.shape[0]
mean = gmm_model.means_
variance = gmm_model.covariances_
weight = np.log(gmm_model.weights_)
calc = np.zeros((n, K))
ind = np.zeros((n, d))
soft = calc
add = np.zeros((n,))
dim = np.zeros((n,))
X_pred = ind
    
ind = np.where(data_matrix != 0, 1, 0)            
dim = np.sum(ind, axis=1)

for i in range(n):
    for j in range(K):
        res = data_matrix[i] - mean[j]
        res = np.multiply(res, ind[i])
        #Multivariate Gaussian
        if j == 0:
            A = (res.T @ inver0) @ res
            C = (dim[i]/2)*np.log(2*np.pi) + np.log(deter0 + 1e-16)/2
        else:
            A = (res.T @ inver1) @ res
            C = (dim[i]/2)*np.log(2*np.pi) + np.log(deter1 + 1e-16)/2
        B = 2
        calc[i, j] = weight[j] + (-A/B) - C

add = logsumexp(calc, axis = 1)

#Since the entire computation is done in log-domain to avoid Numerical instability
#we need to bring it back in its original domain
soft = np.exp(np.subtract(np.transpose(calc), add))

lg = np.sum(add)
    
X_calc = np.transpose(soft) @ gmm_model.means_

#We will use predicted value if the entry is 0 in original rating matrix
data_matrix_pred_GMM = np.where(data_matrix == 0, X_calc, data_matrix)

for i in range(data_matrix_pred_GMM.shape[0]):
    for j in range(data_matrix_pred_GMM.shape[1]):
        data_matrix_pred_GMM[i, j] = round(data_matrix_pred_GMM[i, j])

#For measuring the performance we have to use the predicted matrix
for i in range(X_calc.shape[0]):
    for j in range(X_calc.shape[1]):
        X_pred[i, j] = round(X_calc[i, j])
print("Original Rating Matrix: \n", data_matrix)
print("Rating Matrix After Applying GMM: \n", data_matrix_pred_GMM)
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix != 0, 1, 0)

x = np.multiply(X_pred, ind_matrix)
RMSE_GMM = np.sqrt(np.mean((x - data_matrix)**2))
print("RMSE of GMM Model is %f." %RMSE_GMM)
# Understanding Non-Negative Matrix Factorization(NMF)
from sklearn.decomposition import NMF
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve

X = np.array([[1, 2, 3], [5, 10, 15]])
print("X is:\n", X)
model = NMF(n_components=2, init='random', random_state=42)
W = model.fit_transform(X)
H = model.components_
print("W is:\n", W)
print("H is:\n", H)
print("The Result of Matrix Multiplication of W and H is Same as X:\n", np.matmul(W, H))
Image("../input/inputimage/1.JPG")
Image("../input/inputimage/2.JPG")
Image("../input/inputimage/3.JPG")
model = NMF(n_components=2, init='random', random_state=42)
user_vec = model.fit_transform(data_matrix)
item_vec = model.components_.T

def implicit_ALS(ratings, user_vec, item_vec, lambda_val, iteration, typ):                 
    
    ctr = 1

    if typ == 'user':
        while ctr <= iteration:
            YTY = item_vec.T.dot(item_vec)
            lambdaI = np.eye(YTY.shape[0]) * lambda_val

            for u in range(user_vec.shape[0]):
                user_vec[u, :] = solve((YTY + lambdaI), 
                                        ratings[u, :].dot(item_vec))
            ctr += 1

        return user_vec
    
    if typ == 'item':
        while ctr <= iteration:
            XTX = user_vec.T.dot(user_vec)
            lambdaI = np.eye(XTX.shape[0]) * lambda_val
            
            for i in range(item_vec.shape[0]):
                item_vec[i, :] = solve((XTX + lambdaI), 
                                        ratings[:, i].T.dot(user_vec))
            ctr += 1
        return item_vec
        
    
user_vec = implicit_ALS(data_matrix, user_vec, item_vec, lambda_val=0.2,
                        iteration=20, typ='user')
item_vec = implicit_ALS(data_matrix, user_vec, item_vec, lambda_val=0.2,
                        iteration=20, typ='item')

def predict_all():
        """ Predict ratings for every user and item. """
        predictions = np.zeros((user_vec.shape[0], 
                                item_vec.shape[0]))
        for u in range(user_vec.shape[0]):
            for i in range(item_vec.shape[0]):
                predictions[u, i] = predict(u, i)
                
        return predictions
def predict(u, i):
    """ Single user and item prediction. """
    return user_vec[u, :].dot(item_vec[i, :].T)

predict = predict_all()


data_matrix_pred_ALS = np.where(data_matrix == 0, predict, data_matrix)

for i in range(data_matrix_pred_ALS.shape[0]):
    for j in range(data_matrix_pred_ALS.shape[1]):
        data_matrix_pred_ALS[i, j] = round(data_matrix_pred_ALS[i, j])

#For measuring the performance we have to use the predicted matrix
X_pred = np.zeros((nb_users, nb_movies))
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        X_pred[i, j] = round(predict[i, j])
print("Original Rating Matrix: \n", data_matrix)
print("Rating Matrix After Applying ALS: \n", data_matrix_pred_ALS)
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix != 0, 1, 0)

x = np.multiply(X_pred, ind_matrix)
RMSE_ALS = np.sqrt(np.mean((x - data_matrix)**2))
print("RMSE of ALS Model is %f." %RMSE_ALS)
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
data_matrix_torch = torch.FloatTensor(data_matrix)
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def prediction(self, x):
        pred = self.forward(x)
        return pred.detach().numpy()
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0
    for id_user in range(nb_users):
        input = Variable(data_matrix_torch[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s +=1
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str((train_loss/s).item()))
predict_SAE = np.zeros((nb_users, nb_movies))
for id_user in range(nb_users):
    input = Variable(data_matrix_torch[id_user]).unsqueeze(0)
    predict_SAE[id_user] = sae.prediction(input)

#We will use predicted value if the entry is 0 in original rating matrix
data_matrix_pred_SAE = np.where(data_matrix == 0, predict_SAE, data_matrix)

for i in range(data_matrix_pred_SAE.shape[0]):
    for j in range(data_matrix_pred_SAE.shape[1]):
        data_matrix_pred_SAE[i, j] = round(data_matrix_pred_SAE[i, j])

#For measuring the performance we have to use the predicted matrix
X_pred = np.zeros((nb_users, nb_movies))
for i in range(predict_SAE.shape[0]):
    for j in range(predict_SAE.shape[1]):
        X_pred[i, j] = round(predict_SAE[i, j])
print("Original Rating Matrix: \n", data_matrix)
print("Rating Matrix after Applying Stacked Auto-Encoder: \n", data_matrix_pred_SAE)
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix != 0, 1, 0)

x = np.multiply(X_pred, ind_matrix)
RMSE_SAE = np.sqrt(np.mean((x - data_matrix)**2))
print("RMSE of SAE Model is %f." %RMSE_SAE)
m_cols = ['Movie_ID', 'Title', 'Release_Date', 'Video_Release_Date', 'IMDB_URL']
movies = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep='|',
                     names=m_cols, usecols=range(5),encoding='latin-1')
movies.head(10)
movie_id = movies[movies['Title'] == 'Richard III (1995)']['Movie_ID'].values.item()
print("Movie ID is:", movie_id)
#Create an indicator matrix to ensure the movie was not rated previously
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix == 0, 1, 0)

#Multiply predicted rating matrix with this indicator matrix to consider only
#the predicted ones
pred = np.multiply(data_matrix_pred_SAE, ind_matrix)
pred = pred[:, 9]
pred_df = pd.DataFrame(pred)
pred_df.columns = ['Rating']
pred_df = pred_df[pred_df['Rating'] >= 4]
pred_df = pred_df.head(5)
pred_df
user_id = [3, 4, 5, 6, 8]
users_recommend = users[users['User_ID'].isin(user_id)]
users_recommend
users.tail()
#Create an indicator matrix to ensure the movie was not rated previously
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix == 0, 1, 0)

#Multiply predicted rating matrix with this indicator matrix to consider
#only the predicted ones
pred = np.multiply(data_matrix_pred_SAE, ind_matrix)
pred = pred[939, :]
pred_df = pd.DataFrame(pred)
pred_df.columns = ['Rating']
pred_df = pred_df[pred_df['Rating'] >= 4]
pred_df = pred_df.head(5)
pred_df
movie_id = [1, 2, 10, 11, 15]
movie_recommend = movies[movies['Movie_ID'].isin(movie_id)]
movie_recommend