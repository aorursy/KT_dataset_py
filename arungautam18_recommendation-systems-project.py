import numpy as np
import pandas as pd
import seaborn as sns
import os
from surprise import KNNWithMeans
from surprise import Dataset,Reader
from surprise.model_selection import train_test_split
from collections import defaultdict
from surprise import SVD
from surprise import accuracy
from scipy.sparse.linalg import svds
import copy
# print(os.listdir("../input"))
user_rating_data = pd.read_csv('../input/ratings_Electronics.csv')
user_rating_data.columns = ['UserID','ProductID','Ratings','Timestamp']
user_rating_data.head()
user_rating_data.hist()
# Crate a subset of data with users who has given ratings more than 50 times
users_filter = user_rating_data['UserID'].value_counts()
usersfilter_df = pd.DataFrame(users_filter).reset_index()
usersfilter_df.columns = ['UserID','Count']
usersfilter_df_IDs = usersfilter_df[usersfilter_df['Count'] > 100]['UserID']
user_rating_data_Subset = user_rating_data[user_rating_data['UserID'].isin(usersfilter_df_IDs)]
user_rating_data_Subset
# Data can be more filtered  to have product which are atleast rated 10 times
# products_filter = user_rating_data_Subset['ProductID'].value_counts()
# products_filter_df = pd.DataFrame(products_filter).reset_index()
# products_filter_df.columns = ['ProductID','Count']
# products_filter_df_IDs = products_filter_df[products_filter_df['Count'] > 10]['ProductID']
# user_product_rating_data_Subset = user_rating_data_Subset[user_rating_data_Subset['ProductID'].isin(products_filter_df_IDs)]
# user_product_rating_data_Subset
user_rating_data_Subset.isna().sum()
# user_product_rating_data_Subset.isna().sum()
# group by data according to product and count of users gave ratings
countProductUsers = user_rating_data_Subset.groupby(['ProductID']).agg({'UserID': 'count'}).reset_index()
countProductUsers.columns = ['ProductID','UserID_Counts']
countProductUsers.head()
# group by data according to product and avg rating users has given
avgProductRating = user_rating_data_Subset.groupby(['ProductID']).agg({'Ratings': 'mean'}).reset_index()
avgProductRating.columns = ['ProductID','AvgRating']
avgProductRating.head()
product_rating_set = countProductUsers.merge(avgProductRating,on='ProductID')
product_rating_set
# Sorting data and get top results
product_rating_set.sort_values(["UserID_Counts", "AvgRating"], ascending = (False, False)).head()
# product_rating_set.sort_values(["AvgRating"], ascending = False).head(10)
# applying weightage to get hybrid popularity products to recommend
w1 = 0.8
w2 = 0.4

# score = w1* v1 + w2 *v2 /(w1 + w2)
product_rating_set['Score'] = (w1*(product_rating_set['UserID_Counts']) + w2*(product_rating_set['AvgRating']))/(w1+w2)

product_rating_set.sort_values('Score',ascending = False).head()
# Apply KNNwith means algo to get recommendation based on neighbours
reader = Reader(rating_scale=(0,5))
data = Dataset.load_from_df(user_rating_data_Subset[['UserID','ProductID','Ratings']],reader)
trainset,testset = train_test_split(data, test_size=.3)
trainset.ur
# Apply user user collabrative model
algo = KNNWithMeans(k=50,sim_options={'name':'pearson_baseline','user_based':True})
algo.fit(trainset)
test_predictions_KNN = algo.test(testset)
test_predictions_KNN
# check prediction of one user who has not bought one perticula rproduct
user_rating_data_Subset.loc[(user_rating_data_Subset['UserID'] == 'A1V3TRGWOMA8LC') & ( user_rating_data_Subset['ProductID'] == '0594481813')]
# Product user A1V3TRGWOMA8LC have not bought 0594481813 yet 

pred = algo.predict('A1V3TRGWOMA8LC','0594481813',verbose=True)
# We can see maximum users does not have neighbours or very less number of it in our predictions data
# We can not recommend products on basis of thse less neighbou found data
# Implmenting SVD algo
trainset_svd  = data.build_full_trainset()
len(trainset_svd.ur)
algo_svd = SVD()
algo_svd.fit(trainset_svd)
# prepare test data set
testset_svd = trainset_svd.build_anti_testset()
len(testset_svd)
predictions_svd = algo_svd.test(testset_svd)
predictions_svd
# valuate the model
#Check for accuracy for collabrative model
print("User-based Model : Test Set")
accuracy.rmse(predictions_svd, verbose=True)
top_n = defaultdict(list)
for uid, iid, true_r, est, _ in predictions_svd:
        top_n[uid].append((iid,true_r,est,_))
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
top_n = get_top_n(predictions_svd, n=5)
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
# predict for sample user
pred = algo_svd.predict('A1V3TRGWOMA8LC','B003ES5ZUU',verbose=True)
# Implementing SVDs using matrix generation
# Create pivot table
user_ratings_matrix = user_rating_data_Subset.pivot(index='UserID', columns='ProductID', values='Ratings')
# fill with 0 for the combination which has not been bought
user_ratings_matrix.fillna(0,inplace=True)
U,sigma,Vt = svds(user_ratings_matrix,k=50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = user_ratings_matrix.columns)
preds_df.head()
user_ratings_matrix.reset_index(inplace=True)
userid_col = user_ratings_matrix['UserID']
pred_df_col = preds_df.join(userid_col)
# 'A1V3TRGWOMA8LC',:'0594481813']
pred_df_col[pred_df_col['UserID'] == 'A1V3TRGWOMA8LC']['B003ES5ZUU']
# get top n recommendation using SVD matrix 
# get user and product data which user has not bought yet
# merge predicted rating from the predicition matrix
# /sort and fetch top records

user_id = 'A1V3TRGWOMA8LC'
user_id_pred =  pred_df_col[pred_df_col['UserID'] == user_id].T
user_id_pred.reset_index(inplace =True)
user_id_pred.columns= ['ProductID','Ratings']
user_id_pred.drop(index =  len(user_id_pred) -1,inplace=True)
user_id_pred.sort_values(by='Ratings',ascending=False).head(5)
