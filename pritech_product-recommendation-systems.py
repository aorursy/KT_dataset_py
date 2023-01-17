# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np  

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from collections import defaultdict

from surprise import KNNWithMeans

from surprise import SVD, SVDpp

from surprise import KNNBaseline

from surprise import KNNBasic

from surprise import KNNWithZScore

from surprise import BaselineOnly

from surprise import Dataset

from surprise import Reader

from surprise import accuracy

from surprise.model_selection import train_test_split

from surprise.model_selection import cross_validate

from surprise.model_selection import KFold

from surprise.model_selection import GridSearchCV



import time
start_time = time.time()



df = pd.read_csv("/kaggle/input/ratings-electronics/ratings_Electronics.csv", names=["userId", "productId", "rating", "timestamp"])  

df.head() 



computational_time = time.time() - start_time

print('Done in %0.3fs' %(computational_time))
rows_count, columns_count = df.shape

print('Total Number of rows :', rows_count)

print('Total Number of columns :', columns_count)
df.dtypes
unique_userId = df['userId'].nunique()

unique_productId = df['productId'].nunique()

print('Total number of unique Users    : ', unique_userId)

print('Total number of unique Products : ', unique_productId)
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
df.apply(lambda x : sum(x.isnull()))
df.isnull().sum()
df.isna().any()
df_transpose = df.describe().T

df_transpose
df_transpose[['min', '25%', '50%', '75%', 'max']]
plt.figure(figsize=(20,5))

sns.boxplot(data=df, orient='h', palette='Set2', dodge=False)
start_time = time.time()



sns.pairplot(df, diag_kind= 'kde')



computational_time = time.time() - start_time

print('Done in %0.3fs' %(computational_time))
df['rating'].value_counts()
rating_counts = pd.DataFrame(df['rating'].value_counts()).reset_index()

rating_counts.columns = ['Labels', 'Ratings']

rating_counts
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

sns.countplot(df['rating'], ax=ax1)

ax1.set_xlabel('Rating Distribution', fontsize=10)

ax1.set_ylabel('Count', fontsize=10)





explode = (0.1, 0, 0.1, 0, 0)

ax2.pie(rating_counts["Ratings"], explode=explode, labels=rating_counts.Labels, autopct='%1.2f%%',

        shadow=True, startangle=70)

ax2.axis('equal')

plt.title("Rating Ratio")

plt.legend(rating_counts.Labels, loc=3)

plt.show()
df.corr()
mask = np.zeros_like(df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(5,2))

plt.title('Correlation of Attributes', y=1.05, size=10)

sns.heatmap(df.corr(),vmin=-1, cmap='plasma',annot=True,  mask=mask, fmt='.2f')
df = df.drop(['timestamp'], axis=1)
df1 = df.copy()
df1.head()
users_counts = df1['userId'].value_counts().rename('users_counts')

users_data   = df1.merge(users_counts.to_frame(),

                                left_on='userId',

                                right_index=True)
subset_df = users_data[users_data.users_counts >= 50]

subset_df.head()
product_rating_counts = subset_df['productId'].value_counts().rename('product_rating_counts')

product_rating_data   = subset_df.merge(product_rating_counts.to_frame(),

                                left_on='productId',

                                right_index=True)
product_rating_data = product_rating_data[product_rating_data.product_rating_counts >= 10]

product_rating_data.head()
amazon_df = product_rating_data.copy()
panda_data = amazon_df.drop(['users_counts', 'product_rating_counts'], axis=1)
panda_data.head()
k = 5
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(panda_data[['userId', 'productId', 'rating']], reader)
trainset, testset = train_test_split(surprise_data, test_size=.30, random_state=7)
panda_data.groupby('productId')['rating'].mean().head()
panda_data.groupby('productId')['rating'].mean().sort_values(ascending=False).head()
prod_rating_count = pd.DataFrame(panda_data.groupby('productId')['rating'].mean().sort_values(ascending=False))

prod_rating_count['prod_rating_count'] = pd.DataFrame(panda_data.groupby('productId')['rating'].count())

prod_rating_count.head(k)
basic_poplurity_model = prod_rating_count.sort_values(by=['prod_rating_count'], ascending=False)

basic_poplurity_model.head(k)
#Count of user_id for each unique song as recommendation score 

panda_data_grouped = panda_data.groupby('productId').agg({'userId': 'count'}).reset_index()

panda_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

panda_data_grouped.head()

#Sort the songs on recommendation score 

panda_data_sort = panda_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 

      

#Generate a recommendation rank based upon score 

panda_data_sort['Rank'] = panda_data_sort['score'].rank(ascending=0, method='first') 

          

#Get the top 5 recommendations 

popularity_recommendations = panda_data_sort.head(k) 

popularity_recommendations 
# UsINNG popularity based recommender model to make predictions

import warnings

warnings.filterwarnings('ignore')

def recommend(userId):     

    user_recommendations = popularity_recommendations 

          

    #Adding user_id column for which the recommendations are being generated 

    user_recommendations['userID'] = userId 

      

    #Bringing user_id column to the front 

    cols = user_recommendations.columns.tolist() 

    cols = cols[-1:] + cols[:-1] 

    user_recommendations = user_recommendations[cols] 

          

    return user_recommendations 
find_recom = [15,121,55,230,344]   # This list is user choice.

for i in find_recom:

    print("Here is the recommendation for the userId: %d\n" %(i))

    print(recommend(i))    

    print("\n") 
cv_results = []  # to store cross validation result 
svd_param_grid = {'n_epochs': [20, 25], 'lr_all': [0.007, 0.009, 0.01], 'reg_all': [0.4, 0.6]}



svd_gs = GridSearchCV(SVD, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)

svdpp_gs = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)



svd_gs.fit(surprise_data)

svdpp_gs.fit(surprise_data)



# best RMSE score

print(svd_gs.best_score['rmse'])

print(svdpp_gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(svd_gs.best_params['rmse'])

print(svdpp_gs.best_params['rmse'])
start_time = time.time()



# Creating Model using best parameters

svd_model = SVD(n_epochs=20, lr_all=0.005, reg_all=0.2)



# Training the algorithm on the trainset

svd_model.fit(trainset)





# Predicting for test set

predictions_svd = svd_model.test(testset)



# Evaluating RMSE, MAE of algorithm SVD on 5 split(s) by cross validation

svd_cv = cross_validate(svd_model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



# Storing Crossvalidation Results in dataframe

svd_df = pd.DataFrame.from_dict(svd_cv)

svd_described = svd_df.describe()

cv_results = pd.DataFrame([['SVD', svd_described['test_rmse']['mean'], svd_described['test_mae']['mean'], 

                           svd_described['fit_time']['mean'], svd_described['test_time']['mean']]],

                            columns = ['Model', 'RMSE', 'MAE', 'Fit Time', 'Test Time'])





# get RMSE

print("\n\n==================== Model Evaluation ===============================")

accuracy.rmse(predictions_svd, verbose=True)

print("=====================================================================")

computational_time = time.time() - start_time

print('\n Computational Time : %0.3fs' %(computational_time))

cv_results
start_time = time.time()



# Creating Model using best parameters

svdpp_model = SVDpp(n_epochs=25, lr_all=0.01, reg_all=0.4)



# Training the algorithm on the trainset

svdpp_model.fit(trainset)





# Predicting for test set

predictions_svdpp = svdpp_model.test(testset)



# Evaluating RMSE, MAE of algorithm SVDpp on 5 split(s) by cross validation

svdpp_cv = cross_validate(svdpp_model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



# Storing Crossvalidation Results in dataframe

svdpp_df = pd.DataFrame.from_dict(svdpp_cv)

svdpp_described = svdpp_df.describe()

svdpp_cv_results = pd.DataFrame([['SVDpp', svdpp_described['test_rmse']['mean'], svdpp_described['test_mae']['mean'], 

                           svdpp_described['fit_time']['mean'], svdpp_described['test_time']['mean']]],

                            columns = ['Model', 'RMSE', 'MAE', 'Fit Time', 'Test Time'])



cv_results = cv_results.append(svdpp_cv_results, ignore_index=True)



# get RMSE

print("\n\n==================== Model Evaluation ===============================")

accuracy.rmse(predictions_svdpp, verbose=True)

print("=====================================================================")

computational_time = time.time() - start_time

print('\n Computational Time : %0.3fs' %(computational_time))

cv_results
start_time = time.time()



knn_param_grid = {'bsl_options': {'method': ['als', 'sgd'],

                              'reg': [1, 2]},

              'k': [15, 20, 25, 30, 40, 50, 60],

              'sim_options': {'name': ['msd', 'cosine', 'pearson_baseline']}

              }



knnbasic_gs = GridSearchCV(KNNBasic, knn_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)

knnmeans_gs = GridSearchCV(KNNWithMeans, knn_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)

knnz_gs     = GridSearchCV(KNNWithZScore, knn_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)





knnbasic_gs.fit(surprise_data)

knnmeans_gs.fit(surprise_data)

knnz_gs.fit(surprise_data)



# best RMSE score

print(knnbasic_gs.best_score['rmse'])

print(knnmeans_gs.best_score['rmse'])

print(knnz_gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(knnbasic_gs.best_params['rmse'])

print(knnmeans_gs.best_params['rmse'])

print(knnz_gs.best_params['rmse'])



computational_time = time.time() - start_time

print('\nComputational Time : %0.3fs' %(computational_time))
start_time = time.time()



# Creating Model using best parameters

knnBasic_model = KNNBasic(k=50, sim_options={'name': 'cosine', 'user_based': False})



# Training the algorithm on the trainset

knnBasic_model.fit(trainset)



# Predicting for test set

prediction_knnBasic = knnBasic_model.test(testset)



# Evaluating RMSE, MAE of algorithm KNNBasic on 5 split(s)

knnBasic_cv = cross_validate(knnBasic_model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



# Storing Crossvalidation Results in dataframe

knnBasic_df = pd.DataFrame.from_dict(knnBasic_cv)

knnBasic_described = knnBasic_df.describe()

knnBasic_cv_results = pd.DataFrame([['KNNBasic', knnBasic_described['test_rmse']['mean'], knnBasic_described['test_mae']['mean'], 

                           knnBasic_described['fit_time']['mean'], knnBasic_described['test_time']['mean']]],

                            columns = ['Model', 'RMSE', 'MAE', 'Fit Time', 'Test Time'])



cv_results = cv_results.append(knnBasic_cv_results, ignore_index=True)



# get RMSE

print("\n\n==================== Model Evaluation ===============================")

accuracy.rmse(prediction_knnBasic, verbose=True)

print("=====================================================================")



computational_time = time.time() - start_time

print('\n Computational Time : %0.3fs' %(computational_time))

cv_results

start_time = time.time()



# Creating Model using best parameters

knnZscore_model = KNNWithZScore(k=60, sim_options={'name': 'cosine', 'user_based': False})



# Training the algorithm on the trainset

knnZscore_model.fit(trainset)



# Predicting for testset

prediction_knnZscore = knnZscore_model.test(testset)



# Evaluating RMSE, MAE of algorithm KNNWithZScore on 5 split(s)

knnZscore_cv = cross_validate(knnZscore_model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



# Storing Crossvalidation Results in dataframe

knnZscore_df = pd.DataFrame.from_dict(knnZscore_cv)

knnZscore_described = knnZscore_df.describe()

knnZscore_cv_results = pd.DataFrame([['KNNWithZScore', knnZscore_described['test_rmse']['mean'], knnZscore_described['test_mae']['mean'], 

                           knnZscore_described['fit_time']['mean'], knnZscore_described['test_time']['mean']]],

                            columns = ['Model', 'RMSE', 'MAE', 'Fit Time', 'Test Time'])



cv_results = cv_results.append(knnZscore_cv_results, ignore_index=True)



# get RMSE

print("\n\n==================== Model Evaluation ===============================")

accuracy.rmse(prediction_knnZscore, verbose=True)

print("=====================================================================")



computational_time = time.time() - start_time

print('\n Computational Time : %0.3fs' %(computational_time))

cv_results

start_time = time.time()



# Creating Model using best parameters

knnMeansUU_model = KNNWithMeans(k=60, sim_options={'name': 'cosine', 'user_based': True})



# Training the algorithm on the trainset

knnMeansUU_model.fit(trainset)



# Predicting for testset

prediction_knnMeansUU = knnMeansUU_model.test(testset)



# Evaluating RMSE, MAE of algorithm KNNWithMeans User-User on 5 split(s)

knnMeansUU_cv = cross_validate(knnMeansUU_model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



# Storing Crossvalidation Results in dataframe

knnMeansUU_df = pd.DataFrame.from_dict(knnMeansUU_cv)

knnMeansUU_described = knnMeansUU_df.describe()

knnMeansUU_cv_results = pd.DataFrame([['KNNWithMeans User-User', knnMeansUU_described['test_rmse']['mean'], knnMeansUU_described['test_mae']['mean'], 

                           knnMeansUU_described['fit_time']['mean'], knnMeansUU_described['test_time']['mean']]],

                            columns = ['Model', 'RMSE', 'MAE', 'Fit Time', 'Test Time'])



cv_results = cv_results.append(knnMeansUU_cv_results, ignore_index=True)



# get RMSE

print("\n\n==================== Model Evaluation ===============================")

accuracy.rmse(prediction_knnMeansUU, verbose=True)

print("=====================================================================")



computational_time = time.time() - start_time

print('\n Computational Time : %0.3fs' %(computational_time))

cv_results

start_time = time.time()



# Creating Model using best parameters

knnMeansII_model = KNNWithMeans(k=60, sim_options={'name': 'cosine', 'user_based': False})



# Training the algorithm on the trainset

knnMeansII_model.fit(trainset)



# Predicting for testset

prediction_knnMeansII = knnMeansII_model.test(testset)



# Evaluating RMSE, MAE of algorithm KNNWithMeans Item-Item on 5 split(s)

knnMeansII_cv = cross_validate(knnMeansII_model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



# Storing Crossvalidation Results in dataframe

knnMeansII_df = pd.DataFrame.from_dict(knnMeansII_cv)

knnMeansII_described = knnMeansII_df.describe()

knnMeansII_cv_results = pd.DataFrame([['KNNWithMeans Item-Item', knnMeansII_described['test_rmse']['mean'], knnMeansII_described['test_mae']['mean'], 

                           knnMeansII_described['fit_time']['mean'], knnMeansII_described['test_time']['mean']]],

                            columns = ['Model', 'RMSE', 'MAE', 'Fit Time', 'Test Time'])



cv_results = cv_results.append(knnMeansII_cv_results, ignore_index=True)



# get RMSE

print("\n\n==================== Model Evaluation ===============================")

accuracy.rmse(prediction_knnMeansII, verbose=True)

print("=====================================================================")



computational_time = time.time() - start_time

print('\n Computational Time : %0.3fs' %(computational_time))

cv_results

x_algo = ['KNN Basic', 'KNNWithMeans-User-User', 'KNNWithMeans-Item-Item', 'KNN ZScore', 'SVD', 'SVDpp']

all_algos_cv = [knnBasic_cv, knnMeansUU_cv, knnMeansII_cv, knnZscore_cv, svd_cv, svdpp_cv]



rmse_cv = [round(res['test_rmse'].mean(), 4) for res in all_algos_cv]

mae_cv  = [round(res['test_mae'].mean(), 4) for res in all_algos_cv]



plt.figure(figsize=(20,15))



plt.subplot(2, 1, 1)

plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)

plt.plot(x_algo, rmse_cv, label='RMSE', color='darkgreen', marker='o')

plt.xlabel('Algorithms', fontsize=15)

plt.ylabel('RMSE Value', fontsize=15)

plt.legend()

plt.grid(ls='dashed')



plt.subplot(2, 1, 2)

plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)

plt.plot(x_algo, mae_cv, label='MAE', color='navy', marker='o')

plt.xlabel('Algorithms', fontsize=15)

plt.ylabel('MAE Value', fontsize=15)

plt.legend()

plt.grid(ls='dashed')



plt.show()



cv_results
top_n = defaultdict(list)

def get_top_n(predictions, n=k):

    # First map the predictions to each user.

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:

        top_n[uid].append((iid, est))



    # Then sort the predictions for each user and retrieve the k highest ones.

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n



top_n = get_top_n(predictions_svd, n=k)

top_n
def precision_recall_at_k(predictions, k=5, threshold=3.5):

    '''Return precision and recall at k metrics for each user.'''



    # First map the predictions to each user.

    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:

        user_est_true[uid].append((est, true_r))



    precisions = dict()

    recalls = dict()

    for uid, user_ratings in user_est_true.items():



        # Sort user ratings by estimated value

        user_ratings.sort(key=lambda x: x[0], reverse=True)



        # Number of relevant items

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)



        # Number of recommended items in top k

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])



        # Number of relevant and recommended items in top k

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))

                              for (est, true_r) in user_ratings[:k])



        # Precision@K: Proportion of recommended items that are relevant

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1



        # Recall@K: Proportion of relevant items that are recommended

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1



    return precisions, recalls





kf = KFold(n_splits=5)

svd_model = SVD(n_epochs=20, lr_all=0.005, reg_all=0.2)

precs = []

recalls = []



for trainset, testset in kf.split(surprise_data):

    svd_model.fit(trainset)

    predictions = svd_model.test(testset)

    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3.5)



    # Precision and recall can then be averaged over all users

    print('Precision : ', sum(prec for prec in precisions.values()) / len(precisions))

    print('recalls : ',sum(rec for rec in recalls.values()) / len(recalls))
