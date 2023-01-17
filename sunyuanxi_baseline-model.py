from surprise import Reader, Dataset, SVD, evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=r_cols, index_col='movie_id', encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure',\
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror',\
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('../input/ml-100k/u.item', sep='|', names=m_cols, index_col=0, encoding='latin-1')

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../input/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1', parse_dates=True)
from datetime import datetime

ratings['unix_timestamp'] = ratings['unix_timestamp'].apply(datetime.fromtimestamp)
ratings.columns = ['user_id', 'rating', 'time']
ratings.head(10)
ratings['rating'].hist(bins=9)
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies.head(10)
for i in users['occupation'].unique():
    users[i] = users['occupation'] == i
users.drop('occupation', axis=1, inplace=True)
users.head(10)
ratings_movie_summary = ratings.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
ratings_movie_summary.head(10)
ratings_user_summary = ratings.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
ratings_user_summary.head(10)
ratings_movie_summary.sort_values(by='count')['count'].hist(bins=20)
ratings_movie_summary.sort_values(by='mean')['mean'].hist(bins=20)
ratings_user_summary.sort_values(by='count')['count'].hist(bins=20)
ratings_user_summary.sort_values(by='mean')['mean'].hist(bins=20)
ratings_p = pd.pivot_table(ratings, values='rating', index='user_id', columns='movie_id')
ratings_p.iloc[:10, :10]
mean = ratings_p.stack().mean()
std = ratings_p.stack().std()
movie_mean = np.ones(ratings_p.shape)
movie_mean = pd.DataFrame(movie_mean * np.array(ratings_movie_summary['mean']).reshape(1,1682))
user_mean = np.ones(ratings_p.T.shape)
user_mean = pd.DataFrame(user_mean * np.array(ratings_user_summary['mean'])).T
pred = movie_mean + user_mean - mean
score = abs(np.array(ratings_p) - pred)
score_2 = score ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2.stack().mean())))
print('MAE: {:.4f}'.format(score.stack().mean()))
from sklearn.model_selection import KFold

kfolds = KFold(n_splits = 5, random_state = 13)
rmse = []
mae = []
i = 0
print('Evaluating RMSE, MAE of the Baseline Model. \n')
print('-'*12)
for train_index, test_index in kfolds.split(ratings):
    train = ratings.copy()
    test = ratings.copy()
    train['rating'].iloc[test_index] = np.NaN
    test['rating'].iloc[train_index] = np.NaN
    train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
    train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
    test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)
    movie_mean = np.ones(ratings_p.shape)
    movie_mean = pd.DataFrame(movie_mean * np.array(train_movie_summary['mean']).reshape(1,1682))
    user_mean = np.ones(ratings_p.T.shape)
    user_mean = pd.DataFrame(user_mean * np.array(train_user_summary['mean'])).T
    train_p = movie_mean + user_mean - mean
    score = abs(np.array(test_p) - train_p)
    score_2 = score ** 2
    rmse += [np.sqrt(score_2.stack().mean())]
    mae += [score.stack().mean()]
    i += 1
    print('Fold', i)
    print('RMSE: {:.4f}'.format(np.sqrt(score_2.stack().mean())))
    print('MAE: {:.4f}'.format(score.stack().mean()))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse)))
print('Mean MAE: {:.4f}'.format(np.mean(mae)))
print('-'*12)
print('-'*12)
movie_mean = np.ones(ratings_p.shape)
movie_mean = pd.DataFrame(movie_mean * np.array(ratings_movie_summary['mean']).reshape(1,1682))
user_mean = np.ones(ratings_p.T.shape)
user_mean = pd.DataFrame(user_mean * np.array(ratings_user_summary['mean'])).T
user_std = np.ones(ratings_p.T.shape)
user_std = pd.DataFrame(user_std * np.array(ratings_user_summary['std'])).T
pred_plus = user_mean + (movie_mean - mean)/std * user_std
score_plus = abs(np.array(ratings_p) - pred_plus)
score_2_plus = score_plus ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_plus.stack().mean())))
print('MAE: {:.4f}'.format(score_plus.stack().mean()))
user_196 = movies[['title', 'release_date']]
user_196['Estimate_Score'] = np.array(pred_plus.loc[195])
user_196 = user_196.sort_values('Estimate_Score', ascending=False)
print(user_196.head(10))
rmse_plus = []
mae_plus = []
i = 0
print('Evaluating RMSE, MAE of the Baseline_Plus Model. \n')
print('-'*12)
for train_index, test_index in kfolds.split(ratings):
    train = ratings.copy()
    test = ratings.copy()
    train['rating'].iloc[test_index] = np.NaN
    test['rating'].iloc[train_index] = np.NaN
    train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
    train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
    test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)
    movie_mean = np.ones(ratings_p.shape)
    movie_mean = pd.DataFrame(movie_mean * np.array(train_movie_summary['mean']).reshape(1,1682))
    user_mean = np.ones(ratings_p.T.shape)
    user_mean = pd.DataFrame(user_mean * np.array(train_user_summary['mean'])).T
    user_std = np.ones(ratings_p.T.shape)
    user_std = pd.DataFrame(user_std * np.array(train_user_summary['std'])).T
    train_p = user_mean + (movie_mean - mean)/std * user_std
    score = abs(np.array(test_p) - train_p)
    score_2 = score ** 2
    rmse_plus += [np.sqrt(score_2.stack().mean())]
    mae_plus += [score.stack().mean()]
    i += 1
    print('Fold', i)
    print('RMSE: {:.4f}'.format(np.sqrt(score_2.stack().mean())))
    print('MAE: {:.4f}'.format(score.stack().mean()))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse_plus)))
print('Mean MAE: {:.4f}'.format(np.mean(mae_plus)))
print('-'*12)
print('-'*12)
from sklearn.svm import SVR

movie_mean = np.ones(ratings_p.shape)
movie_mean = pd.DataFrame(movie_mean * np.array(ratings_movie_summary['mean']).reshape(1,1682))
X = np.array(ratings_p*0) + movie_mean
svm = SVR(gamma=1, C=1)
pred_svm = ratings_p.copy()
for i in range(ratings_p.shape[0]):
    svm.fit(np.array(X.iloc[i].dropna()).reshape(-1,1), ratings_p.iloc[i].dropna())
    pred_svm.iloc[i] = svm.predict(np.array(movie_mean.iloc[0]).reshape(-1,1))
score_svm = abs(np.array(ratings_p) - pred_svm)
score_2_svm = score_svm ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_svm.stack().mean())))
print('MAE: {:.4f}'.format(score_svm.stack().mean()))
user_196_svm = movies[['title', 'release_date']]
user_196_svm['Estimate_Score'] = np.array(pred_svm.loc[195])
user_196_svm = user_196_svm.sort_values('Estimate_Score', ascending=False)
print(user_196_svm.head(10))
rmse_svm = []
mae_svm = []
fold = 0
movie_mean = pd.DataFrame(np.ones(ratings_p.shape) * np.array(ratings_movie_summary['mean']).reshape(1,1682))
print('Evaluating RMSE, MAE of the Baseline_SVM Model. \n')
print('-'*12)
for train_index, test_index in kfolds.split(ratings):
    train = ratings.copy()
    test = ratings.copy()
    train['rating'].iloc[test_index] = np.NaN
    test['rating'].iloc[train_index] = np.NaN
    train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
    train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
    train_p = pd.pivot_table(train, values='rating', index='user_id', columns='movie_id', dropna=False)
    test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)
    train_mean = pd.DataFrame(np.ones(ratings_p.shape) * np.array(train_movie_summary['mean']).reshape(1,1682))
    X = np.array(train_p*0) + train_mean
    pred = ratings_p.copy()
    for i in range(ratings_p.shape[0]):
        svm.fit(np.array(X.iloc[i].dropna()).reshape(-1,1), train_p.iloc[i].dropna())
        pred.iloc[i] = svm.predict(np.array(movie_mean.iloc[0]).reshape(-1,1))
    score = abs(np.array(test_p) - pred)
    score_2 = score ** 2
    rmse_svm += [np.sqrt(score_2.stack().mean())]
    mae_svm += [score.stack().mean()]
    fold += 1
    print('Fold', fold)
    print('RMSE: {:.4f}'.format(np.sqrt(score_2.stack().mean())))
    print('MAE: {:.4f}'.format(score.stack().mean()))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse_svm)))
print('Mean MAE: {:.4f}'.format(np.mean(mae_svm)))
print('-'*12)
print('-'*12)
from xgboost import XGBRegressor

movie_mean = np.ones(ratings_p.shape)
movie_mean = pd.DataFrame(movie_mean * np.array(ratings_movie_summary['mean']).reshape(1,1682))
X = np.array(ratings_p*0) + movie_mean
xgb = XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=10, gamma=1)
pred_xgb = ratings_p.copy()
for i in range(ratings_p.shape[0]):
    xgb.fit(np.array(X.iloc[i].dropna()).reshape(-1,1), ratings_p.iloc[i].dropna())
    pred_xgb.iloc[i] = xgb.predict(np.array(movie_mean.iloc[0]).reshape(-1,1))
score_xgb = abs(np.array(ratings_p) - pred_xgb)
score_2_xgb = score_xgb ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_xgb.stack().mean())))
print('MAE: {:.4f}'.format(score_xgb.stack().mean()))
rmse_xgb = []
mae_xgb = []
fold = 0
movie_mean = pd.DataFrame(np.ones(ratings_p.shape) * np.array(ratings_movie_summary['mean']).reshape(1,1682))
print('Evaluating RMSE, MAE of the Baseline_XGB Model. \n')
print('-'*12)
for train_index, test_index in kfolds.split(ratings):
    train = ratings.copy()
    test = ratings.copy()
    train['rating'].iloc[test_index] = np.NaN
    test['rating'].iloc[train_index] = np.NaN
    train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
    train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
    train_p = pd.pivot_table(train, values='rating', index='user_id', columns='movie_id', dropna=False)
    test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)
    train_mean = pd.DataFrame(np.ones(ratings_p.shape) * np.array(train_movie_summary['mean']).reshape(1,1682))
    X = np.array(train_p*0) + train_mean
    pred = ratings_p.copy()
    for i in range(ratings_p.shape[0]):
        xgb.fit(np.array(X.iloc[i].dropna()).reshape(-1,1), train_p.iloc[i].dropna())
        pred.iloc[i] = xgb.predict(np.array(movie_mean.iloc[0]).reshape(-1,1))
    score = abs(np.array(test_p) - pred)
    score_2 = score ** 2
    rmse_xgb += [np.sqrt(score_2.stack().mean())]
    mae_xgb += [score.stack().mean()]
    fold += 1
    print('Fold', fold)
    print('RMSE: {:.4f}'.format(np.sqrt(score_2.stack().mean())))
    print('MAE: {:.4f}'.format(score.stack().mean()))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse_xgb)))
print('Mean MAE: {:.4f}'.format(np.mean(mae_xgb)))
print('-'*12)
print('-'*12)
def recommend(movie_title, min_count):
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = movies[movies['title'] == movie_title].index[0]
    target = ratings_p[i]
    similar_to_target = ratings_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(movies).join(ratings_movie_summary)\
                  [['PearsonR', 'title', 'count', 'mean']]
    print (corr_target[corr_target['count']>min_count][:10].to_string(index=False))
recommend('Shawshank Redemption, The (1994)', 10)
sim = ratings_p.corr().abs()
sim.iloc[:10, :10]
knn_pred = ratings_p.copy()
for i in ratings_p.index:
    N = sim.loc[ratings[ratings['user_id'] == i].index]
    for j in ratings_p.columns:
        try:
            N_k = N[j].sort_values(ascending=False).drop(j)[:30]
        except:
            N_k = N[j].sort_values(ascending=False)[:30]
        weighted_rating = N_k*ratings_p.loc[i, N_k.index]
        knn_pred.loc[i, j] = weighted_rating.sum()/N_k.sum()

knn_pred.iloc[:10, :10]
score_knn = abs(np.array(ratings_p) - knn_pred)
score_2_knn = score_knn ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_knn.stack().mean())))
print('MAE: {:.4f}'.format(score_knn.stack().mean()))
train = ratings.copy()
test = ratings.copy()
train['rating'].iloc[80000:] = np.NaN
test['rating'].iloc[:80000] = np.NaN
train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
train_p = pd.pivot_table(train, values='rating', index='user_id', columns='movie_id', dropna=False)
test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)

knn_pred = ratings_p.copy()
for i in ratings_p.index:
    if i % 100 == 0:
        print(i)
    N = sim.loc[train[train['user_id'] == i].index]
    for j in ratings_p.columns:
        try:
            N_k = N[j].sort_values(ascending=False).drop(j)[:30]
        except:
            N_k = N[j].sort_values(ascending=False)[:30]
        weighted_rating = N_k*train_p.loc[i, N_k.index]
        knn_pred.loc[i, j] = weighted_rating.sum()/N_k.sum()

score_knn = abs(np.array(ratings_p) - knn_pred)
score_2_knn = score_knn ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_knn.stack().mean())))
print('MAE: {:.4f}'.format(score_knn.stack().mean()))
knn_plus_pred = ratings_p.copy()
for i in ratings_p.index:
    N = sim.loc[ratings[ratings['user_id'] == i].index]
    for j in ratings_p.columns:
        try:
            N_k = N[j].sort_values(ascending=False).drop(j)[:30]
        except:
            N_k = N[j].sort_values(ascending=False)[:30]
        weighted_rating = N_k*(ratings_p.loc[i, N_k.index] - ratings_movie_summary.loc[N_k.index, 'mean'])/ ratings_movie_summary.loc[N_k.index, 'std']
        knn_plus_pred.loc[i, j] = weighted_rating.sum()/N_k.sum() * ratings_movie_summary.loc[j, 'std'] + ratings_movie_summary.loc[j, 'mean']

knn_plus_pred.iloc[:10, :10]
score_knn_plus = abs(np.array(ratings_p) - knn_plus_pred)
score_2_knn_plus = score_knn ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_knn_plus.stack().mean())))
print('MAE: {:.4f}'.format(score_knn_plus.stack().mean()))
train = ratings.copy()
test = ratings.copy()
train['rating'].iloc[80001:] = np.NaN
test['rating'].iloc[:80001] = np.NaN
train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
train_p = pd.pivot_table(train, values='rating', index='user_id', columns='movie_id', dropna=False)
test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)

knn_plus_pred = ratings_p.copy()
for i in ratings_p.index:
    if i % 100 == 0:
        print(i)
    N = sim.loc[train[train['user_id'] == i].index]
    for j in ratings_p.columns:
        try:
            N_k = N[j].sort_values(ascending=False).drop(j)[:30]
        except:
            N_k = N[j].sort_values(ascending=False)[:30]
        weighted_rating = N_k*(train_p.loc[i, N_k.index] - train_movie_summary.loc[N_k.index, 'mean'])/ train_movie_summary.loc[N_k.index, 'std']
        knn_plus_pred.loc[i, j] = weighted_rating.sum()/N_k.sum() * train_movie_summary.loc[j, 'std'] + train_movie_summary.loc[j, 'mean']

score_knn = abs(np.array(ratings_p) - knn_plus_pred)
score_2_knn = score_knn ** 2
print('RMSE: {:.4f}'.format(np.sqrt(score_2_knn.stack().mean())))
print('MAE: {:.4f}'.format(score_knn.stack().mean()))
from datetime import datetime

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings['unix_timestamp'] = ratings['unix_timestamp'].apply(datetime.fromtimestamp)
ratings.columns = ['user_id', 'movie_id', 'rating', 'time']
ratings.head(10)

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure',\
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror',\
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('../input/ml-100k/u.item', sep='|', names=m_cols, encoding='latin-1')
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
#occupation = {'none': 0, 'administrator': 1, 'artist': 2, 'doctor': 3, 'educator': 4, 'engineer': 5, 'entertainment': 6,\
#              'executive': 7, 'healthcare': 8, 'homemaker': 9, 'lawyer': 10, 'librarian': 11, 'marketing': 12,\
#              'programmer': 13, 'salesman': 14, 'scientist': 15, 'student': 16, 'technician': 17, 'writer': 18,\
#              'retired': 19, 'other': 20}
df = ratings_p.stack(dropna=False).reset_index()
df.columns = ['user_id', 'movie_id', 'rating']
df = df.merge(users, on='user_id')
df = df.merge(movies, on='movie_id')
df['sex'] = df['sex'].replace(['F', 'M'], [1, 0])
#df['occupation'] = df['occupation'].replace(occupation)
df.drop(['release_date', 'video_release_date', 'imdb_url', 'title', 'zip_code'], axis=1, inplace=True)
df_train = df.dropna()
df.head(10)
rmse_reg = []
mae_reg = []
i = 0
print('Evaluating RMSE, MAE of the XGB_Reg Model. \n')
print('-'*12)
for train_index, test_index in kfolds.split(ratings):
    X_train = df_train.drop('rating', axis=1).iloc[train_index]
    y_train = df_train['rating'].iloc[train_index]
    X_test = df_train.drop('rating', axis=1).iloc[test_index]
    y_test = df_train['rating'].iloc[test_index]
    xgb = XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=10, gamma=0.03).fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    score = abs(y_test - y_pred)
    score_2 = score**2
    rmse_reg += [np.sqrt(np.mean(score_2))]
    mae_reg += [np.mean(score)]
    i += 1
    print('Fold', i)
    print('RMSE: {:.4f}'.format(np.sqrt(np.mean(score_2))))
    print('MAE: {:.4f}'.format(np.mean(score)))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse_reg)))
print('Mean MAE: {:.4f}'.format(np.mean(mae_reg)))
print('-'*12)
print('-'*12)
### BUG. Need Fix. ###

#df = df.merge(ratings_p, left_on='user_id', right_index=True)
#df = df.merge(ratings_p.T, left_on='movie_id', right_index=True)
#df.head(10)
#df_train = df_train.merge(ratings_p, left_on='user_id', right_index=True)
#df_train = df_train.merge(ratings_p.T, left_on='movie_id', right_index=True)
#df_train.head(10)
#for i in range(df_train.shape[0]):
#    if i % 1000 == 0:
#        print(i)
#    row = df_train.iloc[i]
#    df_train.iloc[i][str(row[1])+'_x'] = np.NaN
#    df_train.iloc[i][str(row[0])+'_y'] = np.NaN
#X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['rating'], axis=1), df_train['rating'], random_state = 0)
#xgb = XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=10, gamma=0.03).fit(X_train, y_train)
#y_pred = xgb.predict(X_test)
#score = abs(y_test - y_pred)
#score_2 = score**2
#print('RMSE: {:.4f}'.format(np.sqrt(np.mean(score_2))))
#print('MAE: {:.4f}'.format(np.mean(score)))

# RMSE: 0.7401
# MAE: 0.5674
#pred_196 = df[df['user_id']==196]
#pred_196 = pred_196.merge(ratings_p, left_on='user_id', right_index=True)
#pred_196 = pred_196.merge(ratings_p.T, left_on='movie_id', right_index=True)
#pred_196.head(10)
#xgb = XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=10, gamma=0.03)\
#      .fit(df_train.drop(['rating'], axis=1), df_train['rating'])
#pred_196['rating'] = xgb.predict(pred_196.drop('rating', axis=1))
#user_196_reg = movies[['movie_id', 'title', 'release_date']]
#user_196_reg['Estimate_Score'] = np.array(pred_196['rating'])
#user_196_reg.drop('movie_id', axis=1, inplace=True)
#user_196_reg = user_196_reg.sort_values('Estimate_Score', ascending=False)
#print(user_196_reg.head(10))
#rmse_reg_plus = []
#mae_reg_plus = []
#i = 0
#print('Evaluating RMSE, MAE of the XGB_Reg_Plus Model. \n')
#print('-'*12)
#for train_index, test_index in kfolds.split(ratings):
#    X_train = df_train.drop('rating', axis=1).iloc[train_index]
#    y_train = df_train['rating'].iloc[train_index]
#    X_test = df_train.drop('rating', axis=1).iloc[test_index]
#    y_test = df_train['rating'].iloc[test_index]
#    xgb = XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=10, gamma=0.03).fit(X_train, y_train)
#    y_pred = xgb.predict(X_test)
#    score = abs(y_test - y_pred)
#    score_2 = score**2
#    rmse_reg_plus += [np.sqrt(np.mean(score_2))]
#    mae_reg_plus += [np.mean(score)]
#    i += 1
#    print('Fold', i)
#    print('RMSE: {:.4f}'.format(np.sqrt(np.mean(score_2))))
#    print('MAE: {:.4f}'.format(np.mean(score)))
#    print('-'*12)
#print('-'*12)
#print('Mean RMSE: {:.4f}'.format(np.mean(rmse_reg_plus)))
#print('Mean MAE: {:.4f}'.format(np.mean(mae_reg_plus)))
#print('-'*12)
#print('-'*12)