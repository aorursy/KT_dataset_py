import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from surprise import KNNWithMeans

from surprise import Dataset

from surprise import accuracy

from surprise.model_selection import train_test_split as surprise_train_test_split

from sklearn.model_selection import train_test_split as train_test_split

from surprise import Reader

import scipy.sparse

from scipy.sparse import csr_matrix

from scipy.sparse.linalg import svds

import warnings; warnings.simplefilter('ignore')
cols=['userId','productId','Rating','timestamp']

data=pd.read_csv("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv",names=cols)
data.head(10)
data.dtypes
data=data.drop(['timestamp'],axis=1)
data.info()
na_values=data.isna().sum()

print(na_values)
data.describe().T
#Find the minimum and maximum ratings

print('Minimum rating is: %d' %(data.Rating.min()))

print('Maximum rating is: %d' %(data.Rating.max()))
sns.countplot(data['Rating'])
print("Total number of records with unique users and products")

print("*"*100)

print("The total number of records in the data-set are:", data.shape[0])

print("The total number of unique users in the data-set are:", len(np.unique(data.userId)))

print("The total number of unique products in the data-set are:", len(np.unique(data.productId)))
data_no_of_ratings_userId=data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
data_no_of_ratings_userId.head()
data_no_of_ratings_productId=data.groupby(by='productId')['Rating'].count().sort_values(ascending=False)  
data_no_of_ratings_productId.head()
#data_subset=data[data.groupby('userId')['userId'].transform('size')>50]

counts = data['userId'].value_counts()

data_subset = data[data['userId'].isin(counts[counts >= 50].index)]
##### data_subset=data_subset[data_subset.groupby('productId')['productId'].transform('size')>50]

counts = data_subset['productId'].value_counts()

data_subset = data_subset[data_subset['productId'].isin(counts[counts >= 100].index)]

data_subset.head()
data_subset.shape
print("Total number of records with unique users and products")

print("*"*100)

print("The total number of records in the data-set are:", data_subset.shape[0])

print("The total number of unique users in the data-set are:", len(np.unique(data_subset.userId)))

print("The total number of unique products in the data-set are:", len(np.unique(data_subset.productId)))
final_ratings_matrix = data_subset.pivot(index = 'userId', columns ='productId', values = 'Rating').fillna(0)

print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)



given_num_of_ratings = np.count_nonzero(final_ratings_matrix)

print('given_num_of_ratings = ', given_num_of_ratings)

possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]

print('possible_num_of_ratings = ', possible_num_of_ratings)

density = (given_num_of_ratings/possible_num_of_ratings)

density *= 100

print ('density: {:4.2f}%'.format(density))
final_ratings_matrix.head()
data_PBR=data_subset
data_PBR.groupby('productId')['Rating'].mean().head(10)
data_PBR.groupby('productId')['Rating'].mean().sort_values(ascending=False).head(10)
data_PBR.groupby('productId')['Rating'].count().sort_values(ascending=False).head(10)
mean_count_ratings=pd.DataFrame(data_PBR.groupby('productId')['Rating'].mean())
mean_count_ratings['Rating counts']=data_PBR.groupby('productId')['Rating'].count()
recommended_products=mean_count_ratings[(mean_count_ratings['Rating']>4.5) & (mean_count_ratings['Rating counts']>50)]
recommended_products
#Split the data randomnly into train and test datasets into 70:30 ratio

train_data, test_data = train_test_split(data_subset, test_size = 0.3, random_state=0)

train_data.head()
print('Shape of training data: ',train_data.shape)

print('Shape of testing data: ',test_data.shape)
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()

train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

train_data_grouped.head()
#Sort the products on recommendation score 

train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 

      

#Generate a recommendation rank based upon score 

train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 

          

#Get the top 5 recommendations 

popularity_recommendations = train_data_sort.head(5) 

popularity_recommendations
def recommend(user_id):     

    user_recommendations = popularity_recommendations 

          

    #Add user_id column for which the recommendations are being generated 

    user_recommendations['userId'] = user_id 

      

    #Bring user_id column to the front 

    cols = user_recommendations.columns.tolist() 

    cols = cols[-1:] + cols[:-1] 

    user_recommendations = user_recommendations[cols] 

          

    return user_recommendations
find_recom = [11,123,290]   # This list is user choice.

for i in find_recom:

    print("The list of recommendations for the userId: %d\n" %(i))

    print(recommend(i))    

    print("\n")
reader = Reader(rating_scale=(1, 5))
data_CFBR=data_subset

data_CFBR = Dataset.load_from_df(data_CFBR[['userId', 'productId', 'Rating']], reader)
trainset, testset = surprise_train_test_split(data_CFBR, test_size=.3)
CF_Model = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})

CF_Model.fit(trainset)
#Prediction on the testset

test_pred = CF_Model.test(testset)
test_pred
#RMSE

print("Item-based Model : Train Set")

accuracy.rmse(test_pred, verbose=True)

CF_Model = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})

CF_Model.fit(trainset)
#Getting predictions

test_pred = CF_Model.test(testset)
test_pred
# get RMSE

print("Item-based Model : Train Set")

accuracy.rmse(test_pred, verbose=True)

pivot_df = data_subset.pivot(index = 'userId', columns ='productId', values = 'Rating').fillna(0)

pivot_df.head()
print('Shape of the pivot table: ', pivot_df.shape)
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)

pivot_df.head()
pivot_df.set_index(['user_index'], inplace=True)

# Actual ratings given by users

pivot_df.head()
# Singular Value Decomposition

U, sigma, Vt = svds(pivot_df, k = 10)
print('Left singular matrix: \n',U)
print('Sigma: \n',sigma)
# Construct diagonal array in SVD

sigma = np.diag(sigma)

print('Diagonal matrix: \n',sigma)
print('Right singular matrix: \n',Vt)
#Predicted ratings

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

# Convert predicted ratings to dataframe

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)

preds_df.head()


def recommend_items(userID, pivot_df, preds_df, num_recommendations):

    # index starts at 0  

    user_idx = userID-1 

    # Get and sort the user's ratings

    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)

    #sorted_user_ratings

    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)

    #sorted_user_predictions

    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)

    temp.index.name = 'Recommended Items'

    temp.columns = ['user_ratings', 'user_predictions']

    temp = temp.loc[temp.user_ratings == 0]   

    temp = temp.sort_values('user_predictions', ascending=False)

    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(userID))

    print(temp.head(num_recommendations))
userID = 50

num_recommendations = 5

recommend_items(userID, pivot_df, preds_df, num_recommendations)
userID = 5

num_recommendations = 5

recommend_items(userID, pivot_df, preds_df, num_recommendations)
userID = 8

num_recommendations = 5

recommend_items(userID, pivot_df, preds_df, num_recommendations)
final_ratings_matrix.head()
final_ratings_matrix.mean().head()
preds_df.head()


preds_df.mean().head()
rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)

rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']

print(rmse_df.shape)

rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)

rmse_df.head()
RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)

print('\nRMSE SVD Model = {} \n'.format(RMSE))
userID = 9

num_recommendations = 5

recommend_items(userID, pivot_df, preds_df, num_recommendations)