# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data_set=pd.read_csv("/kaggle/input/amazon-product-reviews/ratings_Electronics (1).csv",names=['userid', 'productid','rating','timestamp'])

data_set.info()

data_set.shape
data_set_buffer = data_set.drop('timestamp',axis=1)

#consider only 10% of the data

data_set_buffer = data_set_buffer.sample(frac=0.1)

del data_set
data_set_buffer.head().T
#print missing value

print(data_set_buffer.isna().sum())

print(data_set_buffer.isnull().sum())
data_set_buffer.describe().T
data_set_buffer.rating.describe().T
data_set_buffer.groupby('userid')['rating'].count().sort_values(ascending=False)
sns.set(style="white", palette="tab10", color_codes=True)
ax = sns.countplot(data=data_set_buffer,x='rating');

ax.set_ylim(0, len(data_set_buffer))

ax.set_xlim(0, 5)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:%}'.format(height/float(len(data_set_buffer))),

            ha="center") 

plt.show();
data_set_buffer_with_threshold50=data_set_buffer.groupby("productid").filter(lambda x:x['rating'].count() >=50)

del data_set_buffer
data_set_buffer_with_threshold50.groupby('productid')['rating'].count().sort_values(ascending=False)
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data_set_buffer_with_threshold50, test_size = 0.3, random_state=0)

train_data.head()
#Count no of user_id for each unique product as recommendation score 

train_data_grouped = train_data.groupby('productid').agg({'userid': 'count'}).reset_index()

train_data_grouped.rename(columns = {'userid': 'noofusers'},inplace=True)

train_data_grouped.head()
#Count no of user_id for each unique product as recommendation score 

train_data_grouped_rating= train_data.groupby(['productid'])['rating'].sum().reset_index()

train_data_grouped_rating.rename(columns = {'rating': 'ratingsum'},inplace=True)

train_data_grouped_rating.head()
#top five prouducts as per their avg rating

#Count of user_id for each unique product as recommendation score 

train_data_grouped_users = train_data.groupby('productid').agg({'userid': 'count'}).reset_index()

train_data_grouped_users.rename(columns = {'userid': 'noofuser'},inplace=True)

train_data_grouped_users.head()
train_data_merged_grouped = pd.merge(train_data_grouped_rating, train_data_grouped_users, on='productid')

train_data_merged_grouped.head()
train_data_merged_grouped['averagerating']= train_data_merged_grouped['ratingsum']/train_data_merged_grouped['noofuser']

train_data_merged_grouped.head()
train_data_merged_grouped.sort_values('averagerating',ascending=False)
# Find top 5 popular products

train_data_merged_grouped.sort_values('averagerating',ascending=False).head(5)
del train_data_grouped

del train_data_grouped_rating

del train_data_merged_grouped
from surprise import KNNWithMeans

from surprise import Dataset

from surprise import accuracy

from surprise import Reader

from surprise.model_selection import train_test_split
#Load the dataframe to surprise. Observation : Got memory error so considering almost 1% only of the original dataset!!

data_set_buffer_with_threshold50 = data_set_buffer_with_threshold50.sample(frac=0.1)

data = Dataset.load_from_df(data_set_buffer_with_threshold50,Reader(rating_scale=(1, 5)))

trainset, testset = train_test_split(data, test_size=.30)
# Use user_based true/false to switch between user-based or item-based collaborative filtering

userusercollaborativefiltering = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': True})

userusercollaborativefiltering.fit(trainset)
trainset.n_users
test_pred = userusercollaborativefiltering.test(testset)
#RMSE

print("User-based Model : Test Set RMSE score")

accuracy.rmse(test_pred, verbose=True)
# item-based collaborative filtering

itembasedcollaborativefiltering = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})

itembasedcollaborativefiltering.fit(trainset)
test_pred_I = itembasedcollaborativefiltering.test(testset)
#RMSE

print("Irem-based Model : Test Set RMSE score")

accuracy.rmse(test_pred_I, verbose=True)
del test_pred_I

del trainset

del data
from collections import defaultdict

from surprise import SVD
# First train an SVD algorithm with dataset.data_set_buffer = data_set_buffer.sample(frac=0.1)

dataset_svd = data_set_buffer_with_threshold50.sample(frac=0.01)

dataset_svd = Dataset.load_from_df(dataset_svd,Reader(rating_scale=(1, 5)))
trainset = dataset_svd.build_full_trainset()

svd_algo = SVD()

svd_algo.fit(trainset)
#Predict ratings for all pairs (u, i) that are NOT in the training set.

testset = trainset.build_anti_testset()
predictions = svd_algo.test(testset)
def get_top_n_recommendations(reccomemndations, n=5):

    # First map the reccommendations to each user.

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in reccomemndations:

        top_n[uid].append((iid, est))



    #sort predictions for each user and retrieve the k highest ones.

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n
top_5 = get_top_n_recommendations(predictions, n=5)
for uid, user_ratings in top_5.items():

    print(uid, [iid for (iid, _) in user_ratings])