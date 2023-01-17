# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import math

import json

import time

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split

from sklearn.neighbors import NearestNeighbors

#from sklearn.externals import joblib

import scipy.sparse

from scipy.sparse import csr_matrix

import warnings; warnings.simplefilter('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import the data set

df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')

df.head()
# Dropping the columns

df = df.drop(['Id', 'ProfileName','Time','HelpfulnessNumerator','HelpfulnessDenominator','Text','Summary'], axis = 1) 

# see few rows of the imported dataset

df.tail()
# Check the number of rows and columns

rows, columns = df.shape

print("No of rows: ", rows) 

print("No of columns: ", columns) 
#Check Data types

df.dtypes
# Check for missing values present

print('Number of missing values across columns-\n', df.isnull().sum())
# Summary statistics of 'rating' variable

df[['Score']].describe().transpose()
# find minimum and maximum ratings 



def find_min_max_rating():

    print('The minimum rating is: %d' %(df['Score'].min()))

    print('The maximum rating is: %d' %(df['Score'].max()))

    

find_min_max_rating() 
# Check the distribution of ratings 

with sns.axes_style('white'):

    g = sns.factorplot("Score", data=df, aspect=2.0,kind='count')

    g.set_ylabels("Total number of ratings")
# Number of unique user id and product id in the data

print('Number of unique USERS in Raw data = ', df['UserId'].nunique())

print('Number of unique ITEMS in Raw data = ', df['ProductId'].nunique())
# Top 10 users based on rating

most_rated = df.groupby('UserId').size().sort_values(ascending=False)[:10]

most_rated
counts = df['UserId'].value_counts()

df_final = df[df['UserId'].isin(counts[counts >= 50].index)]
df_final.head()
print('Number of users who have rated 50 or more items =', len(df_final))

print('Number of unique USERS in final data = ', df_final['UserId'].nunique())

print('Number of unique ITEMS in final data = ', df_final['ProductId'].nunique())


final_ratings_matrix = pd.pivot_table(df_final,index=['UserId'], columns = 'ProductId', values = "Score")

final_ratings_matrix.fillna(0,inplace=True)

print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

given_num_of_ratings = np.count_nonzero(final_ratings_matrix)

print('given_num_of_ratings = ', given_num_of_ratings)

possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]

print('possible_num_of_ratings = ', possible_num_of_ratings)

density = (given_num_of_ratings/possible_num_of_ratings)

density *= 100

print ('density: {:4.2f}%'.format(density))
final_ratings_matrix.tail()
# Matrix with one row per 'Product' and one column per 'user' for Item-based CF

final_ratings_matrix_T = final_ratings_matrix.transpose()

final_ratings_matrix_T.head()
#Split the training and test data in the ratio 70:30

train_data, test_data = train_test_split(df_final, test_size = 0.3, random_state=0)



print(train_data.head(5))
def shape():

    print("Test data shape: ", test_data.shape)

    print("Train data shape: ", train_data.shape)

shape() 
#Count of user_id for each unique product as recommendation score 

train_data_grouped = train_data.groupby('ProductId').agg({'UserId': 'count'}).reset_index()

train_data_grouped.rename(columns = {'UserId': 'score'},inplace=True)

train_data_grouped.head()
#Sort the products on recommendation score 

train_data_sort = train_data_grouped.sort_values(['score', 'ProductId'], ascending = [0,1]) 

      

#Generate a recommendation rank based upon score 

train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first') 

          

#Get the top 5 recommendations 

popularity_recommendations = train_data_sort.head(5) 

popularity_recommendations 
# Use popularity based recommender model to make predictions

def recommend(user_id):     

    user_recommendations = popularity_recommendations 

          

    #Add user_id column for which the recommendations are being generated 

    user_recommendations['UserId'] = user_id 

      

    #Bring user_id column to the front 

    cols = user_recommendations.columns.tolist() 

    cols = cols[-1:] + cols[:-1] 

    user_recommendations = user_recommendations[cols] 

          

    return user_recommendations
find_recom = [15,121,200]   # This list is user choice.

for i in find_recom:

    print("Here is the recommendation for the userId: %d\n" %(i))

    print(recommend(i))    

    print("\n") 
print('Since this is a popularity-based recommender model, recommendations remain the same for all users')

print('\nWe predict the products based on the popularity. It is not personalized to particular user')
df_CF = pd.concat([train_data, test_data]).reset_index()

df_CF.tail()
#User-based Collaborative Filtering

# Matrix with row per 'user' and column per 'item' 

pivot_df = pd.pivot_table(df_CF,index=['UserId'], columns = 'ProductId', values = "Score")

pivot_df.fillna(0,inplace=True)

print(pivot_df.shape)

pivot_df.head()
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)

pivot_df.head()
pivot_df.set_index(['user_index'], inplace=True)



# Actual ratings given by users

pivot_df.head()
from scipy.sparse.linalg import svds

# Singular Value Decomposition

U, sigma, Vt = svds(pivot_df, k = 50)

# Construct diagonal array in SVD

sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 



# Predicted ratings

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)

preds_df.head()
# Recommend the items with the highest predicted ratings



def recommend_items(userID, pivot_df, preds_df, num_recommendations):

      

    user_idx = userID-1 # index starts at 0

    

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
#Enter 'userID' and 'num_recommendations' for the user #

userID = 121

num_recommendations = 5

recommend_items(userID, pivot_df, preds_df, num_recommendations)
# Actual ratings given by the users

final_ratings_matrix.head()
# Average ACTUAL rating for each item

final_ratings_matrix.mean().head()
# Predicted ratings 

preds_df.head()
# Average PREDICTED rating for each item

preds_df.mean().head()
rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)

rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']

print(rmse_df.shape)

rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)

rmse_df.head()
RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)

print('\nRMSE SVD Model = {} \n'.format(RMSE))
# Enter 'userID' and 'num_recommendations' for the user #

userID = 200

num_recommendations = 5

recommend_items(userID, pivot_df, preds_df, num_recommendations)