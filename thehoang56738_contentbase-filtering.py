import pandas as pd 



u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('../input/ml-100k/u.user', sep='|', names=u_cols,

encoding='latin-1')



n_users = users.shape[0]

print('Number of users:', n_users)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']



ratings_base = pd.read_csv('../input/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

ratings_test = pd.read_csv('../input/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')



rate_train = ratings_base.as_matrix()

rate_test = ratings_test.as_matrix()

print(ratings_base[0:20])



print('Number of traing rates:', rate_train.shape[0])

print('Number of test rates:', rate_test.shape[0])
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',

 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',

 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']



items = pd.read_csv('../input/ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')



n_items = items.shape[0]

print("Number of items:",n_items)
X0 = items.as_matrix()

#Thể loại phim ở 19 phần tử cuối X_train_counts là ma trận 1682x19 1682 bộ phim và 19 thể loại phim

X_train_counts = X0[:, -19:]

print(X_train_counts.shape)
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=True, norm ='l2')

tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

print(X_train_counts[0])

#Mỗi hàng là các profile của các item (các bộ phim)

print(tfidf[0])
import numpy as np

def get_items_rated_by_user(rate_matrix, user_id):

    """

    return (item_ids, scores)

    """

    y = rate_matrix[:,0] # users

    # Lấy chỉ số đánh giá của user

    # user_id bắt đầu = 1

    ids = np.where(y == user_id +1)[0] 

    item_ids = rate_matrix[ids, 1] - 1 # ids - 1

    scores = rate_matrix[ids, 2]

    return (item_ids, scores)

ids, scores = get_items_rated_by_user(rate_train, 0)

list_head = np.concatenate((ids.reshape(len(ids),1), scores.reshape(len(scores),1)),axis = 1)[0:10]

print(list_head)
from sklearn.linear_model import Ridge # mô hình linear regression với regularization



d = tfidf.shape[1] # data dimension

W = np.zeros((d, n_users))

b = np.zeros((1, n_users))



for n in range(n_users):    

    ids, scores = get_items_rated_by_user(rate_train, n)

    clf = Ridge(alpha=0.01, fit_intercept  = True)

    Xhat = tfidf[ids, :]

    

    clf.fit(Xhat, scores) 

    W[:, n] = clf.coef_

    b[0, n] = clf.intercept_
#model dự đoán

Yhat = tfidf.dot(W) + b 

# print(Yhat[0])
n = 100

ids, scores = get_items_rated_by_user(rate_test, 0)

Yhat[n, ids]

print('Rated movies ids:', ids )

print('True ratings:', scores)

print('Predicted ratings:', Yhat[ids, n])
from math import sqrt

def evaluate(Yhat, rates, W, b):

    se = 0

    cnt = 0

    for n in range(n_users):

        ids, scores_truth = get_items_rated_by_user(rates, n)

        scores_pred = Yhat[ids, n]

        e = scores_truth - scores_pred 

        se += (e*e).sum(axis = 0)

        cnt += e.size 

    return sqrt(se/cnt)

print('RMSE for training:', evaluate(Yhat, rate_train, W, b))

print('RMSE for test    :', evaluate(Yhat, rate_test, W, b))