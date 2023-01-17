
#importing necessary Libraries 

#working with data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection


import sklearn 
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

from collections import defaultdict
from surprise import SVD
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
import os


import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
Data=pd.read_csv("/kaggle/input/amazon-product-reviews/ratings_Electronics (1).csv",names=['UserId', 'ProductId','Rating','timestamp'])

# Display the data

Data.head()

#checking datatypes of each column
Data.dtypes
#shape of data 
shape_Data = Data.shape
print('Data set contains "{x}" number of rows and "{y}" number of columns' .format(x=shape_Data[0],y=shape_Data[1]))
#null check
sns.heatmap(Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Oveview of Data
Data.describe().T
print("Total data ")
print("-"*50)
print("\nTotal no of ratings :",Data.shape[0])
print("Total No of Users   :", len(np.unique(Data['UserId'])))
print("Total No of products  :", len(np.unique(Data['ProductId'])))

# Rating frequency

sns.set(rc={'figure.figsize': (12, 6)})
sns.set_style('whitegrid')
ax = sns.countplot(x='Rating', data=Data)
ax.set(xlabel='Rating', ylabel='Count')
# let's check what is on avarage rating of each product
Rating_prod = Data.groupby('ProductId')['Rating'].mean()
Rating_prod.head()
sns.distplot(Rating_prod, color="green", kde=True)
# let's check how many rating does a product have

product_rating_count = Data.groupby('ProductId')['Rating'].count()
product_rating_count.head()
sns.distplot(product_rating_count, color="red", kde=True, bins=40)
#Analysis of rating given by the user 

no_of_rated_products_per_user = Data.groupby(by='UserId')['Rating'].count().sort_values(ascending=False)
no_of_rated_products_per_user.head()
sns.distplot(no_of_rated_products_per_user, color="Orange", kde=True, bins=40)
# checking number of users how gave 1 rating rating only.
user_1=no_of_rated_products_per_user[no_of_rated_products_per_user==1].count()
#percentage of user who gave rating only one time are
per = user_1/no_of_rated_products_per_user.count()
print('Total {} percent of User have just given rating once'.format(per*100))
print('\n Number of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )
#Getting the new dataframe which contains users who has given 50 or more ratings

new_Data=Data.groupby("ProductId").filter(lambda x:x['Rating'].count() >=50)
new_Data.head()
new_Data.shape
#percentage of data taken
print('we are taking {} percent of data from Raw data for analysis'.format(new_Data['UserId'].count()/Data['UserId'].count()*100))
#Dropping Unwanted Columns
new_Data.drop('timestamp',inplace=True,axis=1)
#group by product and corresponding mean rating
ratings_mean_count = pd.DataFrame(new_Data.groupby('ProductId')['Rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(new_Data.groupby('ProductId')['Rating'].count())
#let's check for highest rating count
ratings_mean_count['rating_counts'].max()
#let's check for highest rating count
ratings_mean_count['rating_counts'].min()
#checking distribution of rating_counts
sns.distplot(ratings_mean_count['rating_counts'],kde=False, bins=40)
#checking distribution of rating
sns.distplot(ratings_mean_count['Rating'],kde=False, bins=40)
#Top 10 Product that would be recommended.
popular=ratings_mean_count.sort_values(['rating_counts','Rating'], ascending=False)
popular.head(10)
#Top 30 Product that would be recommended.
popular.head(30).plot(kind='bar')
#Reading the dataset using Surprise package for Model Based Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data_reader_SVD = Dataset.load_from_df(new_Data,reader)
#Splitting the dataset with 70% training and 30% testing using Surprise train_test_split
trainset_SVD, testset_SVD = train_test_split(data_reader_SVD, test_size=.30)
#Data Split for Memory Based Collaborative Filtering
# we were going out of memory problem so lets take first 10lac record to Collaborative filtering process.
# so splitting data in diffrent part to train them saparately 
# splitting data into 5 Equal parts of 1074862 record each
reader = Reader(rating_scale=(1, 5))
data_reader_1 = Dataset.load_from_df(new_Data.iloc[:1074862,0:],reader)
data_reader_2 = Dataset.load_from_df(new_Data.iloc[1074862:2149725,0:],reader)
data_reader_3 = Dataset.load_from_df(new_Data.iloc[2149725:3224586,0:],reader)
data_reader_4 = Dataset.load_from_df(new_Data.iloc[3224586:4299448,0:],reader)
data_reader_5 = Dataset.load_from_df(new_Data.iloc[4299448:,0:],reader)

#Splitting the dataset with 70% training and 30% testing using Surprise train_test_split
trainset_1, testset_1 = train_test_split(data_reader_1, test_size=.30)
trainset_2, testset_2 = train_test_split(data_reader_2, test_size=.30)
trainset_3, testset_3 = train_test_split(data_reader_3, test_size=.30)
trainset_4, testset_4 = train_test_split(data_reader_4, test_size=.30)
trainset_5, testset_5 = train_test_split(data_reader_5, test_size=.30)

#holding all training set
trainset=[trainset_1,trainset_2,trainset_3,trainset_4,trainset_5]
#holding all testing set
testset=[testset_1,testset_2,testset_3,testset_4,testset_5]
# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
#fitting all training set and storing testing results
test=[]
for item in range(5):
    algo.fit(trainset[item])
    test.append(algo.test(testset[item]))
#checking prediction
test[0][0:5]
algo_SVD = SVD()
algo_SVD.fit(trainset_SVD)
predictions_SVD = algo.test(testset_SVD)
RMSE_SVD=accuracy.rmse(predictions_SVD, verbose=True)
popular=ratings_mean_count.sort_values(['rating_counts','Rating'], ascending=False)
popular.head(10)
# evaluating Collobarative filtering (memory based model)
print("Item-based Model : Test Set")
RMSE = []
Total_RMSE = 0
for i in range(5):
    RMSE.append(accuracy.rmse(test[i], verbose=True))
    Total_RMSE = Total_RMSE + RMSE[i]
#avarage RMSE
print ('Avarage RMSE for Memory Based Collaborative Filtering of all TEST data is = {}'.format(Total_RMSE/5))
# evaluating Collobarative filtering (Model based model)
print ('Avarage RMSE for Model Based Collaborative Filtering of all TEST data is = {}'.format(RMSE_SVD))

#creating function to get top 5 Product Recommendation for each user.
def get_top_n(predictions, n=5):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
top_n = get_top_n(predictions_SVD, n=5)
# Print the recommended items for first 50 user
count=0
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
    if(count>49):
        break
    count=count+1