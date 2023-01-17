import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from surprise import Reader, Dataset, SVD, SVDpp, NMF, KNNBaseline, evaluate, accuracy
from surprise.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure',\
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror',\
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('../input/ml-100k/u.item', sep='|', names=m_cols, encoding='latin-1')

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../input/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1', parse_dates=True)
ratings['unix_timestamp'] = ratings['unix_timestamp'].apply(datetime.fromtimestamp)
ratings.columns = ['user_id', 'movie_id', 'rating', 'time']
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
#trainset, testset = train_test_split(ratings, test_size=0.15, random_state=0)

reader = Reader()
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
#data.split(n_folds=3)

svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
testset = trainset.build_testset()
predictions = svd.test(testset)
model_pred = pd.DataFrame([[i.uid, i.iid, i.est] for i in predictions], columns=['user_id', 'movie_id', 'svd'])
model_pred.shape
anti_testset = trainset.build_anti_testset()
anti_predictions = svd.test(anti_testset)
model_pred_anti = pd.DataFrame([[i.uid, i.iid, i.est] for i in anti_predictions], columns=['user_id', 'movie_id', 'svd'])
model_pred = pd.concat([model_pred, model_pred_anti], ignore_index=True)
model_pred.shape
svd_p = pd.pivot_table(model_pred, values='svd', index='user_id', columns='movie_id')
svd_p.iloc[:10, :10]
svd_p = np.array(svd_p)
u, s, vt = np.linalg.svd(svd_p, full_matrices=False)
sigma = np.diag(s)
print(u.shape, sigma.shape, vt.shape)
pd.DataFrame(sigma[:10, :10])
pd.DataFrame(np.matmul(u, np.matmul(sigma, vt))).iloc[:10, :10]
#from surprise.model_selection import GridSearchCV
#from surprise.model_selection import cross_validate

#param_grid = {'n_factors': [110, 120, 140, 160], 'n_epochs': [90, 100, 110], 'lr_all': [0.001, 0.003, 0.005, 0.008],
#              'reg_all': [0.08, 0.1, 0.15]}
#gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
#gs.fit(data)
#algo = gs.best_estimator['rmse']
#print(gs.best_score['rmse'])
#print(gs.best_params['rmse'])
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
param_grid = {'n_factors': [70, 80, 90, 100, 110, 120, 130, 140, 150, 160], 'n_epochs': [100], 'reg_all': [0.1]}
gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs_svd.fit(data)
svd = gs_svd.best_estimator['rmse']
print(gs_svd.best_score['rmse'])
print(gs_svd.best_params['rmse'])
cv_results_svd = pd.DataFrame(gs_svd.cv_results)
fig = plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(cv_results_svd['mean_test_rmse'])
plt.xticks(np.arange(10), np.arange(70, 170, 10), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('number of latent factors', fontsize=16)
plt.ylabel('Root Mean Square Error (RMSE)', fontsize=16)
plt.grid()
plt.legend()
plt.subplot(122)
plt.plot(cv_results_svd['mean_test_mae'])
plt.xticks(np.arange(10), np.arange(70, 170, 10), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('number of latent factors', fontsize=16)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=16)
plt.grid()
plt.legend()
plt.show()
eval_svd = evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
svd.fit(trainset)
df_196 = ratings[ratings['user_id'] == 196]
df_196 = df_196.set_index('movie_id')
df_196 = df_196.join(movies)['title']
print(df_196.sort_index())
user_196_svd = movies[['movie_id', 'title', 'release_date']]
user_196_svd['Estimate_Score'] = user_196_svd['movie_id'].apply(lambda x: svd.predict('196', x).est)
user_196_svd = user_196_svd.drop('movie_id', axis = 1)
user_196_svd = user_196_svd.sort_values('Estimate_Score', ascending=False)
print(user_196_svd.head(10))
svdpp = SVDpp()
evaluate(svdpp, data, measures=['RMSE', 'MAE'])
param_grid = {'lr_all': [0.001, 0.003, 0.005, 0.007, 0.009], 'reg_all': [0.005, 0.01, 0.015, 0.02, 0.025]}
gs_svdpp = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
gs_svdpp.fit(data)
svdpp = gs_svdpp.best_estimator['rmse']
print(gs_svdpp.best_score['rmse'])
print(gs_svdpp.best_params['rmse'])
cv_results_svdpp = pd.DataFrame(gs_svdpp.cv_results)
svdpp_rmse = np.array(cv_results_svdpp['mean_test_rmse']).reshape(5,5)
svdpp_mae = np.array(cv_results_svdpp['mean_test_mae']).reshape(5,5)
fig = plt.figure(figsize=(15,6))
ax1 = plt.subplot(121)
im1 = ax1.imshow(svdpp_rmse)
cbar = ax1.figure.colorbar(im1, ax=ax1)
ax1.set_xticks(np.arange(5))
ax1.set_yticks(np.arange(5))
ax1.set_xticklabels(param_grid['lr_all'], fontsize=13)
ax1.set_yticklabels(param_grid['reg_all'], fontsize=13)
for i in range(5):
    for j in range(5):
        text = ax1.text(j, i, round(svdpp_rmse[i][j], 4), ha="center", va="center", color="w")
ax1.set_xlabel('reg_all', fontsize=16)
ax1.set_ylabel('lr_all', fontsize=16)
ax1.set_title('Root Mean Square Error (RMSE)', fontsize=16)
ax2 = plt.subplot(122)
im2 = ax2.imshow(svdpp_mae)
cbar = ax2.figure.colorbar(im2, ax=ax2)
ax2.set_xticks(np.arange(5))
ax2.set_yticks(np.arange(5))
ax2.set_xticklabels(param_grid['lr_all'], fontsize=13)
ax2.set_yticklabels(param_grid['reg_all'], fontsize=13)
for i in range(5):
    for j in range(5):
        text = ax2.text(j, i, round(svdpp_mae[i][j], 4), ha="center", va="center", color="w")
ax2.set_xlabel('reg_all', fontsize=16)
ax2.set_ylabel('lr_all', fontsize=16)
ax2.set_title('Mean Absolute Error (MAE)', fontsize=16)
plt.show()
eval_svdpp = evaluate(svdpp, data, measures=['RMSE', 'MAE'])
nmf = NMF()
evaluate(nmf, data, measures=['RMSE', 'MAE'])
param_grid = {'n_factors': [1,2,3,4,5,6,7,8,9,10], 'n_epochs': [100], 'biased': [True], 'reg_bu': [0.1], 'reg_bi': [0.1]}
gs_nmfb = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3)
gs_nmfb.fit(data)
nmfb = gs_nmfb.best_estimator['rmse']
print(gs_nmfb.best_score['rmse'])
print(gs_nmfb.best_params['rmse'])
cv_results_nmfb = pd.DataFrame(gs_nmfb.cv_results)
fig = plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(cv_results_nmfb['mean_test_rmse'])
plt.xticks(np.arange(10), np.arange(1, 11), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('number of latent factors', fontsize=16)
plt.ylabel('Root Mean Square Error (RMSE)', fontsize=16)
plt.grid()
plt.legend()
plt.subplot(122)
plt.plot(cv_results_nmfb['mean_test_mae'])
plt.xticks(np.arange(10), np.arange(1, 11), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('number of latent factors', fontsize=16)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=16)
plt.grid()
plt.legend()
plt.show()
eval_nmfb = evaluate(nmfb, data, measures=['RMSE', 'MAE'])
param_grid = {'n_factors': [200, 220, 240], 'n_epochs': [90, 100, 110]}
gs_nmf = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3)
gs_nmf.fit(data)
nmf = gs_nmf.best_estimator['rmse']
print(gs_nmf.best_score['rmse'])
print(gs_nmf.best_params['rmse'])
eval_nmf = evaluate(nmf, data, measures=['RMSE', 'MAE'])
knnb = KNNBaseline(k=50)
evaluate(knnb, data, measures=['RMSE', 'MAE'])
knnb_1 = KNNBaseline(k=60, sim_options = {'user_based': False})
eval_knnb_1 = evaluate(knnb_1, data, measures=['RMSE', 'MAE'])
param_grid = {'k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'sim_options': {'user_based': [True, False]},\
              'bsl_options': {'method': ['als', 'sgd']}}
gs_knn = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)
gs_knn.fit(data)
print(gs_knn.best_score['rmse'])
print(gs_knn.best_params['rmse'])
cv_results_knn = pd.DataFrame(gs_knn.cv_results)
index = np.arange(0, 40, 4)
fig = plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(cv_results_knn.loc[index, 'mean_test_rmse'].tolist(), label='user_based_als')
plt.plot(cv_results_knn.loc[index+1, 'mean_test_rmse'].tolist(), label='user_based_sgd')
plt.plot(cv_results_knn.loc[index+2, 'mean_test_rmse'].tolist(), label='item_based_als')
plt.plot(cv_results_knn.loc[index+3, 'mean_test_rmse'].tolist(), label='item_based_sgd')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index, 'mean_test_rmse'].tolist(), cv_results_knn.loc[index, 'std_test_rmse'].tolist(), capsize=8, label='user_based_als')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index+1, 'mean_test_rmse'].tolist(), cv_results_knn.loc[index+1, 'std_test_rmse'].tolist(), capsize=8, label='user_based_sgd')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index+2, 'mean_test_rmse'].tolist(), cv_results_knn.loc[index+2, 'std_test_rmse'].tolist(), capsize=8, label='item_based_als')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index+3, 'mean_test_rmse'].tolist(), cv_results_knn.loc[index+3, 'std_test_rmse'].tolist(), capsize=8, label='item_based_sgd')
plt.xticks(np.arange(10), np.arange(10, 110, 10), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('number of neighbors (k)', fontsize=16)
plt.ylabel('Root Mean Square Error (RMSE)', fontsize=16)
plt.grid()
plt.legend()
plt.subplot(122)
plt.plot(cv_results_knn.loc[index, 'mean_test_mae'].tolist(), label='user_based_als')
plt.plot(cv_results_knn.loc[index+1, 'mean_test_mae'].tolist(), label='user_based_sgd')
plt.plot(cv_results_knn.loc[index+2, 'mean_test_mae'].tolist(), label='item_based_als')
plt.plot(cv_results_knn.loc[index+3, 'mean_test_mae'].tolist(), label='item_based_sgd')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index, 'mean_test_mae'].tolist(), cv_results_knn.loc[index, 'std_test_mae'].tolist(), capsize=8, label='user_based_als')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index+1, 'mean_test_mae'].tolist(), cv_results_knn.loc[index+1, 'std_test_mae'].tolist(), capsize=8, label='user_based_sgd')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index+2, 'mean_test_mae'].tolist(), cv_results_knn.loc[index+2, 'std_test_mae'].tolist(), capsize=8, label='item_based_als')
#plt.errorbar(np.arange(10), cv_results_knn.loc[index+3, 'mean_test_mae'].tolist(), cv_results_knn.loc[index+3, 'std_test_mae'].tolist(), capsize=8, label='item_based_sgd')
plt.xticks(np.arange(10), np.arange(10, 110, 10), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('number of neighbors (k)', fontsize=16)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=16)
plt.grid()
plt.legend()
plt.show()
knnb = KNNBaseline(k=70, sim_options = {'user_based': False}, bsl_options = {'method': 'sgd', 'n_epochs': 100})
eval_knnb = evaluate(knnb, data, measures=['RMSE', 'MAE'])
### All models after grid search.
svd = SVD(n_factors=140, n_epochs=100, reg_all=0.1)
eval_svd = evaluate(svd, data, measures=['RMSE', 'MAE'])
svdpp = SVDpp(lr_all=0.005, reg_all=0.015)
eval_svdpp = evaluate(svdpp, data, measures=['RMSE', 'MAE'])
nmfb = NMF(n_factors=3, n_epochs=100, biased=True, reg_bu=0.1, reg_bi=0.1)
eval_nmfb = evaluate(nmfb, data, measures=['RMSE', 'MAE'])
nmf = NMF(n_factors=240, n_epochs=90)
eval_nmf = evaluate(nmf, data, measures=['RMSE', 'MAE'])
knnb_1 = KNNBaseline(k=60, sim_options = {'user_based': False})
eval_knnb_1 = evaluate(knnb_1, data, measures=['RMSE', 'MAE'])
knnb = KNNBaseline(k=70, sim_options = {'user_based': False}, bsl_options = {'method': 'sgd', 'n_epochs': 100})
eval_knnb = evaluate(knnb, data, measures=['RMSE', 'MAE'])
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
trainset = data.build_full_trainset()
testset = trainset.build_testset()
pred = ratings[['user_id', 'movie_id', 'rating']]
l = [svd, svdpp, nmf, nmfb, knnb, knnb_1]
for i in range(len(l)):
    predictions = l[i].test(testset)
    model_pred = pd.DataFrame([[i.uid, i.iid, i.est] for i in predictions], columns=['user_id', 'movie_id', str(i)])
    pred = pd.merge(pred, model_pred, how='left', left_on=['user_id', 'movie_id'], right_on=['user_id', 'movie_id'])
pred.columns = pred.columns[:3].tolist() + ['svd', 'svdpp', 'nmf', 'nmfb', 'knnb', 'knnb_1']
#pred = pd.merge(pred, users, on='user_id')
#pred = pd.merge(pred, movies, on='movie_id')
#pred['sex'] = pred['sex'].replace(['F', 'M'], [1, 0])
#pred.drop(['release_date', 'video_release_date', 'imdb_url', 'title', 'zip_code'], axis=1, inplace=True)
pred['svm'] = np.zeros(pred.shape[0])
for i in pred.index:
    pred.loc[i, 'svm'] = pred_svm.loc[pred.loc[i, 'user_id'], pred.loc[i, 'movie_id']]
pred.head()
linreg = LinearRegression().fit(pred.iloc[:, 3:], pred['rating'])

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
pred['pred'] = linreg.predict(pred.iloc[:, 3:])
print('RMSE: {:.4f}'.format(np.sqrt(((pred['pred'] - pred['rating']) ** 2).mean())))
print('MAE: {:.4f}'.format(abs(pred['pred'] - pred['rating']).mean()))
kf = KFold(n_splits=5, random_state=13)
l = [svd, svdpp, nmf, nmfb, knnb, knnb_1]
predCV = ratings[['user_id', 'movie_id', 'rating']]
predCV_train = ratings[['user_id', 'movie_id', 'rating']]
for trainset, testset in kf.split(data):
    for i in range(len(l)):
        l[i].fit(trainset)
        predictions = l[i].test(testset)
        model_pred = pd.DataFrame([[i.uid, i.iid, i.est] for i in predictions], columns=['user_id', 'movie_id', str(i)])
        predCV = pd.merge(predCV, model_pred, how='outer', left_on=['user_id', 'movie_id'], right_on=['user_id', 'movie_id'])
        testset_train = trainset.build_testset()
        predictions = l[i].test(testset_train)
        model_pred_train = pd.DataFrame([[i.uid, i.iid, i.est] for i in predictions], columns=['user_id', 'movie_id', str(i)])
        predCV_train = pd.merge(predCV_train, model_pred_train, how='outer', left_on=['user_id', 'movie_id'], right_on=['user_id', 'movie_id'])
predCV.columns = ['user_id', 'movie_id', 'rating', 'svd_1', 'svdpp_1', 'nmf_1', 'nmfb_1', 'knnb_1', 'knnb_1_1',\
                  'svd_2', 'svdpp_2', 'nmf_2', 'nmfb_2', 'knnb_2', 'knnb_1_2',\
                  'svd_3', 'svdpp_3', 'nmf_3', 'nmfb_3', 'knnb_3', 'knnb_1_3',\
                  'svd_4', 'svdpp_4', 'nmf_4', 'nmfb_4', 'knnb_4', 'knnb_1_4',\
                  'svd_5', 'svdpp_5', 'nmf_5', 'nmfb_5', 'knnb_5', 'knnb_1_5']
predCV_train.columns = ['user_id', 'movie_id', 'rating', 'svd_1', 'svdpp_1', 'nmf_1', 'nmfb_1', 'knnb_1', 'knnb_1_1',\
                        'svd_2', 'svdpp_2', 'nmf_2', 'nmfb_2', 'knnb_2', 'knnb_1_2',\
                        'svd_3', 'svdpp_3', 'nmf_3', 'nmfb_3', 'knnb_3', 'knnb_1_3',\
                        'svd_4', 'svdpp_4', 'nmf_4', 'nmfb_4', 'knnb_4', 'knnb_1_4',\
                        'svd_5', 'svdpp_5', 'nmf_5', 'nmfb_5', 'knnb_5', 'knnb_1_5']
predCV_train.head()
index = pd.DataFrame(predCV['svd_1'])
index['svd_2'] = predCV['svd_2']
index['svd_3'] = predCV['svd_3']
index['svd_4'] = predCV['svd_4']
index['svd_5'] = predCV['svd_5']
index = index*0+1
index.columns = [1,2,3,4,5]

rmse_svm = []
mae_svm = []
fold = 0
movie_mean = pd.DataFrame(np.ones(ratings_p.shape) * np.array(ratings_movie_summary['mean']).reshape(1,1682))
print('Evaluating RMSE, MAE of the Baseline_SVM Model. \n')
print('-'*12)
for i in index.columns:
    train = ratings.copy()
    test = ratings.copy()
    train['rating'] = train['rating']*(index[i].isna())
    train['rating'].replace(0, np.NaN, inplace=True)
    test['rating'] = test['rating']*index[i]
    train_movie_summary = train.groupby('movie_id')['rating'].agg(['count', 'mean', 'std'])
    train_user_summary = train.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])
    train_p = pd.pivot_table(train, values='rating', index='user_id', columns='movie_id', dropna=False)
    test_p = pd.pivot_table(test, values='rating', index='user_id', columns='movie_id', dropna=False)
    train_mean = pd.DataFrame(np.ones(ratings_p.shape) * np.array(train_movie_summary['mean']).reshape(1,1682))
    X = np.array(train_p*0) + train_mean
    pred = ratings_p.copy()
    for j in range(ratings_p.shape[0]):
        svm.fit(np.array(X.iloc[j].dropna()).reshape(-1,1), train_p.iloc[j].dropna())
        pred.iloc[j] = svm.predict(np.array(movie_mean.iloc[0]).reshape(-1,1))
    predCV['svm_'+str(i)] = np.zeros(predCV.shape[0])
    for x in predCV.index:
        predCV.loc[x, 'svm_'+str(i)] = pred.loc[predCV.loc[x, 'user_id'], predCV.loc[x, 'movie_id']]
    predCV['svm_'+str(i)] = index[i] * predCV['svm_'+str(i)]
    predCV_train['svm_'+str(i)] = np.zeros(predCV.shape[0])
    for x in predCV_train.index:
        predCV_train.loc[x, 'svm_'+str(i)] = pred.loc[predCV_train.loc[x, 'user_id'], predCV_train.loc[x, 'movie_id']]
    predCV_train['svm_'+str(i)] = (index[i].isna()) * predCV_train['svm_'+str(i)]
    predCV_train.replace(0, np.NaN, inplace=True)
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
columns=['user_id', 'movie_id', 'rating', 'svd_1', 'svdpp_1', 'nmf_1', 'nmfb_1', 'knnb_1', 'knnb_1_1', 'svm_1',\
         'svd_2', 'svdpp_2', 'nmf_2', 'nmfb_2', 'knnb_2', 'knnb_1_2', 'svm_2',\
         'svd_3', 'svdpp_3', 'nmf_3', 'nmfb_3', 'knnb_3', 'knnb_1_3', 'svm_3',\
         'svd_4', 'svdpp_4', 'nmf_4', 'nmfb_4', 'knnb_4', 'knnb_1_4', 'svm_4',\
         'svd_5', 'svdpp_5', 'nmf_5', 'nmfb_5', 'knnb_5', 'knnb_1_5', 'svm_5']
predCV=predCV.reindex(columns=columns)
predCV_train=predCV_train.reindex(columns=columns)
coef = linreg.coef_
for i in range(1, 6):
    predCV['fold_'+str(i)] = coef[0] * predCV['svd_'+str(i)] + coef[1] * predCV['svdpp_'+str(i)] +\
                             coef[2] * predCV['nmf_'+str(i)] + coef[3] * predCV['nmfb_'+str(i)] +\
                             coef[4] * predCV['knnb_'+str(i)] + coef[5] * predCV['knnb_1_'+str(i)] +\
                             coef[6] * predCV['svm_'+str(i)] + linreg.intercept_
predCV.head()
rmse_ult = []
mae_ult = []
print('Evaluating RMSE, MAE of the Ultimate Model. \n')
print('-'*12)
for fold in range(1, 6):
    print('Fold', fold)
    rmse_ult += [np.sqrt(((predCV['fold_'+str(fold)] - predCV['rating']) ** 2).mean())]
    mae_ult += [abs(predCV['fold_'+str(fold)] - predCV['rating']).mean()]
    print('RMSE: {:.4f}'.format(np.sqrt(((predCV['fold_'+str(fold)] - predCV['rating']) ** 2).mean())))
    print('MAE: {:.4f}'.format(abs(predCV['fold_'+str(fold)] - predCV['rating']).mean()))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse_ult)))
print('Mean MAE: {:.4f}'.format(np.mean(mae_ult)))
print('-'*12)
print('-'*12)
predCV_1 = predCV.copy()
for i in range(5):
    s = predCV_train.iloc[:, 7*i+3:7*(i+1)+3]
    s['rating'] = predCV_train['rating']
    s.dropna(inplace=True)
    linreg = LinearRegression().fit(s.drop('rating', axis=1), s['rating'])
    coef = linreg.coef_
    predCV_1['fold_'+str(i+1)] = coef[0] * predCV['svd_'+str(i+1)] + coef[1] * predCV['svdpp_'+str(i+1)] +\
                                 coef[2] * predCV['nmf_'+str(i+1)] + coef[3] * predCV['nmfb_'+str(i+1)] +\
                                 coef[4] * predCV['knnb_'+str(i+1)] + coef[5] * predCV['knnb_1_'+str(i+1)] +\
                                 coef[6] * predCV['svm_'+str(i+1)] + linreg.intercept_
rmse_ult = []
mae_ult = []
print('Evaluating RMSE, MAE of the Ultimate Model. \n')
print('-'*12)
for fold in range(1, 6):
    print('Fold', fold)
    rmse_ult += [np.sqrt(((predCV_1['fold_'+str(fold)] - predCV_1['rating']) ** 2).mean())]
    mae_ult += [abs(predCV_1['fold_'+str(fold)] - predCV_1['rating']).mean()]
    print('RMSE: {:.4f}'.format(np.sqrt(((predCV_1['fold_'+str(fold)] - predCV_1['rating']) ** 2).mean())))
    print('MAE: {:.4f}'.format(abs(predCV_1['fold_'+str(fold)] - predCV_1['rating']).mean()))
    print('-'*12)
print('-'*12)
print('Mean RMSE: {:.4f}'.format(np.mean(rmse_ult)))
print('Mean MAE: {:.4f}'.format(np.mean(mae_ult)))
print('-'*12)
print('-'*12)
surprise_results = {'SVD': [np.mean(eval_svd['rmse']), np.mean(eval_svd['mae'])], 'SVDpp': [np.mean(eval_svdpp['rmse']), np.mean(eval_svdpp['mae'])],\
                    'NMF': [np.mean(eval_nmf['rmse']), np.mean(eval_nmf['mae'])], 'Biased_NMF': [np.mean(eval_nmfb['rmse']), np.mean(eval_nmfb['mae'])],\
                    'kNN_SGD': [np.mean(eval_knnb['rmse']), np.mean(eval_knnb['mae'])], 'kNN_ALS': [np.mean(eval_knnb_1['rmse']), np.mean(eval_knnb_1['mae'])],\
                    'Baseline_SVM': [np.mean(rmse_svm), np.mean(mae_svm)], 'Ultimate': [np.mean(rmse_ult), np.mean(mae_ult)]}
surprise_results = pd.DataFrame(surprise_results, index=['RMSE', 'MAE']).T
surprise_results
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
ax.set_axisbelow(True)
plt.bar(np.arange(1, 3*surprise_results.shape[0], 3), surprise_results['RMSE']-0.7, width=1, label='RMSE')
plt.bar(np.arange(2, 3*surprise_results.shape[0], 3), surprise_results['MAE']-0.7, width=1, label='MAE')
plt.xticks(np.arange(1.5, 3*surprise_results.shape[0], 3), surprise_results.index)
plt.yticks(np.arange(0, 0.4, 0.1), [0.7, 0.8, 0.9, 1.0])
plt.grid()
plt.legend()
plt.show()
a = ratings.copy()
a['pred'] = np.zeros(ratings.shape[0])
for i in range(1, 6):
    a['pred'] = a['pred'] + predCV_1['fold_'+str(i)].replace(np.NaN, 0)
pred_1 = a[a['rating']==1]
pred_2 = a[a['rating']==2]
pred_3 = a[a['rating']==3]
pred_4 = a[a['rating']==4]
pred_5 = a[a['rating']==5]
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
hist_4 = plt.hist(pred_4['pred'], bins=100, alpha=0.5, label='4')
hist_3 = plt.hist(pred_3['pred'], bins=100, alpha=0.5, label='3')
hist_5 = plt.hist(pred_5['pred'], bins=100, alpha=0.5, label='5')
hist_2 = plt.hist(pred_2['pred'], bins=100, alpha=0.5, label='2')
hist_1 = plt.hist(pred_1['pred'], bins=100, alpha=0.5, label='1')
handles, labels = ax.get_legend_handles_labels()
handles = [handles.pop(2)]+handles
plt.legend(handles=handles[::-1], title='True Values')
plt.ylabel('Number of Ratings', fontsize=16)
plt.xlabel('Prediction Values', fontsize=16)
plt.title('Prediction Distribution by True Values', fontsize=16)
plt.xlim(0, 6)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
fake_1 = pred[pred['rating']==1]
fake_2 = pred[pred['rating']==2]
fake_3 = pred[pred['rating']==3]
fake_4 = pred[pred['rating']==4]
fake_5 = pred[pred['rating']==5]
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
hist_4 = plt.hist(fake_4['pred'], bins=100, alpha=0.5, label='4')
hist_3 = plt.hist(fake_3['pred'], bins=100, alpha=0.5, label='3')
hist_5 = plt.hist(fake_5['pred'], bins=100, alpha=0.5, label='5')
hist_2 = plt.hist(fake_2['pred'], bins=100, alpha=0.5, label='2')
hist_1 = plt.hist(fake_1['pred'], bins=100, alpha=0.5, label='1')
handles, labels = ax.get_legend_handles_labels()
handles = [handles.pop(2)]+handles
plt.legend(handles=handles[::-1], title='True Values')
plt.ylabel('Number of Ratings', fontsize=16)
plt.xlabel('Prediction Values', fontsize=16)
plt.title('Prediction Distribution by True Values (Without Train Test Split)', fontsize=16)
plt.xlim(0, 6)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()