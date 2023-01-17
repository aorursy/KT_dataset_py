import pandas as pd 
rate_base = pd.read_csv('/kaggle/input/bx-csv-dump/BX-Book-Ratings.csv', sep=';', encoding='latin-1')
rate_base = rate_base[:50000]
rate_base
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
rate_base['User-ID'] = le.fit_transform(rate_base['User-ID'])
rate_base['ISBN'] = le.fit_transform(rate_base['ISBN'])
rated_base = rate_base[rate_base['Book-Rating']!=0]
nonrate_base = rate_base[rate_base['Book-Rating']==0]
print('Dữ liệu đã đánh giá: ', rated_base.shape)
print('Dữ liệu chưa được đánh giá: ', nonrate_base.shape)
from sklearn.model_selection import train_test_split
rated_base_arr = rated_base.values
rate_train, rate_test = train_test_split(rated_base_arr, test_size=0.33, random_state=42)
rate = rate_train.copy()
print ('Dữ liệu train: ', rate_train.shape)
print ('Dữ liệu test: ', rate_test.shape)
import numpy as np
n_ratings = rate_train.shape[0]
n_users = int(np.max(rate_train[:, 0])) + 1
n_items = int(np.max(rate_train[:, 1])) + 1 
print('Số lượng users: ',n_users)
print('Số lượng items: ',n_items)  
import numpy as np
K = 10
lamda = .1
learning_rate = 1
epoch= 100
mu = np.zeros((n_users,))
for n in range(n_users):
    ids = np.where(rate_train[:, 0]  == n)[0]
    item_ids = rate_train[ids, 1] 
    ratings = rate_train[ids, 2]
    # take mean
    m = np.mean(ratings) 
    if np.isnan(m):
        m = 0 
    mu[n] = m
    rate_train[ids, 2] = ratings - mu[n]    
X = np.random.randn(n_items, K)
W = np.random.randn(K, n_users)
def get_items_rated_by_user(user_id):
    ids = np.where(rate_train[:,0] == user_id)[0]
    item_ids = rate_train[ids, 1]
    ratings = rate_train[ids, 2]
    return (item_ids, ratings)
        
        
def get_users_who_rate_item(item_id):
    ids = np.where(rate_train[:,1] == item_id)[0] 
    user_ids = rate_train[ids, 0]
    ratings = rate_train[ids, 2]
    return (user_ids, ratings)
def updateX(X):
    for m in range(n_items):
        user_ids, ratings = get_users_who_rate_item(m)
        Wm = W[:, user_ids]
        # gradient
        grad_xm = (X[m, :].dot(Wm) - ratings).dot(Wm.T)/n_ratings + lamda*X[m, :]
        X[m, :] -= learning_rate*grad_xm.reshape((K,))
    
def updateW(W):
    for n in range(n_users):
        item_ids, ratings = get_items_rated_by_user(n)
        Xn = X[item_ids, :]
        # gradient
        grad_wn = Xn.T.dot(Xn.dot(W[:, n]) - ratings)/n_ratings + lamda*W[:, n]
        W[:, n] -= learning_rate*grad_wn.reshape((K,))
def pred(user, item):
    user = int(user)
    item = int(item)
    bias = mu[user]
    pred = X[item, :].dot(W[:, user]) + bias
    return pred  
def loss(rating, X, W):
        L = 0 
        for i in range(rating.shape[0]):
            # user, item, rating
            n, m, rate = int(rating[i, 0]), int(rating[i, 1]), rating[i, 2]
            L += 0.5*(rate - X[m, :].dot(W[:, n]))**2
        
        L /= rating.shape[0]
        L += 0.5*lamda*(np.linalg.norm(X, 'fro') + np.linalg.norm(W, 'fro'))
        return L 
def evaluate_RMSE(rate):
    n = rate.shape[0]
    SE = 0 # squared error
    for i in range(n):
        predict = pred(rate[i, 0], rate[i, 1])        
        SE += (predict - rate[i, 2])**2 
        RMSE = np.sqrt(SE/n)
    return RMSE
for it in range(epoch):
    updateX(X)
    updateW(W)
    rmse_train = evaluate_RMSE(rate)
    print ('epoch =', it + 1, ', loss =', loss(rate_train, X, W), ', RMSE train =', rmse_train)
X,W
def pred_for_user(data):
    predicted_ratings = np.zeros((data.shape[0],))
    item_ids = np.zeros((data.shape[0],))
    for i in range(data.shape[0]):
        user_id = data[i,0]
        ids = np.where(data[:, 0] == user_id)[0]
        for i in ids:
            predicted_ratings[ids] = round(X[data[i, 1],:].dot(W[:, user_id]) + mu[user_id],2)
    return predicted_ratings
RMSE = evaluate_RMSE(rate_test)
test_table = pd.DataFrame()
test_table['User-ID']=le.inverse_transform(rate_test[:,0])
test_table['ISBN']=le.inverse_transform(rate_test[:,1])
test_table['Book-Rating']=rate_test[:,2]
test_table['Predict-Rating'] = pred_for_user(rate_test)
print('RMSE test:', RMSE)
test_table
nonrate_base_arr = nonrate_base.values
predict_nonrate = pred_for_user(nonrate_base_arr)
RMSE_nonrate = evaluate_RMSE(nonrate_base_arr)
nonrate_table = pd.DataFrame()
nonrate_table['User-ID']=le.inverse_transform(nonrate_base_arr[:,0])
nonrate_table['ISBN']=le.inverse_transform(nonrate_base_arr[:,1])
nonrate_table['Predict-Rating'] = predict_nonrate
print('Phần không được người dùng đánh giá', len(nonrate_table))
print('Phần được dự đoán: ', len(nonrate_table[nonrate_table['Predict-Rating']!=0]))
nonrate_table[nonrate_table['Predict-Rating']!=0]