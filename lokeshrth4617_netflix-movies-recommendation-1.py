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
# Reading dataset file

dataset = pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_1.txt',header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])



# Convert Ratings column to a float

dataset['Rating'] = dataset['Rating'].astype(float)

dataset.head()
#To print the datatype of columns

dataset.dtypes
#To inspect the shape of the datset

dataset.shape
#To print the head of dataset

dataset.head()
#To find the distribution of different ratings in the datset

d = dataset.groupby('Rating')['Rating'].agg(['count'])

d =pd.DataFrame(d)

d
movie_count = dataset['Rating'].isnull().sum()

movie_count
# get customer count

cust_count = dataset['Cust_Id'].nunique()-movie_count



cust_count
# get rating count



rating_count = dataset['Cust_Id'].count() - movie_count

rating_count
import matplotlib.pyplot as plt

ax = d.plot(kind = 'barh', legend = False, figsize = (15,10))

plt.title(f'Total pool: {movie_count} Movies, {cust_count} customers, {rating_count} ratings given', fontsize=20)

plt.axis('On')



for i in range(1,6):

    ax.text(d.iloc[i-1][0]/4, i-1,'Rating {}: {:.0f}%'.format(i, d.iloc[i-1][0]*100 / d.sum()[0]), color = 'white', weight = 'bold')
dataset.head()
# To count all the 'nan' values in the Ratings column in the 'ratings' dataset

df_nan = pd.DataFrame(pd.isnull(dataset.Rating),)



df_nan.head()
df = pd.isnull(dataset['Rating'])

df1 = pd.DataFrame(df)

df2 = df1[df1['Rating']==True]

df2
df2 = df2.reset_index()

df_nan = df2.copy()
df_nan
#To create a numpy array containing movie ids according the 'ratings' dataset



movie_np = []

movie_id = 1



for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):

    # numpy approach

    temp = np.full((1,i-j-1), movie_id)

    movie_np = np.append(movie_np, temp)

    movie_id += 1



# Account for last record and corresponding length

# numpy approach

last_record = np.full((1,len(dataset) - df_nan.iloc[-1, 0] - 1),movie_id)

movie_np = np.append(movie_np, last_record)



#print(f'Movie numpy: {movie_np}')

#print(f'Length: {len(movie_np)}')
#To append the above created array to the datset after removing the 'nan' rows

dataset = dataset[pd.notnull(dataset['Rating'])]



dataset['Movie_Id'] = movie_np.astype(int)

dataset['Cust_Id'] =dataset['Cust_Id'].astype(int)

print('-Dataset examples-')

dataset.head()

f = ['count','mean']
#To create a list of all the movies rated less often(only include top 30% rated movies)

dataset_movie_summary = dataset.groupby('Movie_Id')['Rating'].agg(f)



dataset_movie_summary.index = dataset_movie_summary.index.map(int)



#dataset_movie_summary.index
movie_benchmark = round(dataset_movie_summary['count'].quantile(0.7),0)



#movie_benchmark

drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] < movie_benchmark].index



print('Minimum number of times, a movie should reviewed: {}'.format(movie_benchmark))
#To create a list of all the inactive users(users who rate less often)

dataset_cust_summary = dataset.groupby('Cust_Id')['Rating'].agg(f)



dataset_cust_summary.index = dataset_cust_summary.index.map(int)



cust_benchmark = round(dataset_cust_summary['count'].quantile(0.7),0)



drop_cust_list = dataset_cust_summary[dataset_cust_summary['count'] < cust_benchmark].index



print(f'Customer minimum times of review: {cust_benchmark}')
print(f'Original Shape: {dataset.shape}')

dataset = dataset[~dataset['Movie_Id'].isin(drop_movie_list)]

dataset = dataset[~dataset['Cust_Id'].isin(drop_cust_list)]

print('After Trim Shape: {}'.format(dataset.shape))
print('-Data Examples-')

dataset.head()
df_p = pd.pivot_table(dataset,values='Rating',index='Cust_Id',columns='Movie_Id')



print(df_p.shape)

df_p
df_title = pd.read_csv('/kaggle/input/netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])



df_title.set_index('Movie_Id', inplace = True)



print (df_title.head(10))
# Import required libraries

import math

import re

import matplotlib.pyplot as plt



from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate
# Load Reader library

reader = Reader()



# get just top 100K rows for faster run time

data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)



# Use the SVD algorithm.

svd = SVD()



# Compute the RMSE of the SVD algorithm

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
dataset.head()
dataset_712664 = dataset[(dataset['Cust_Id'] == 712664) & (dataset['Rating'] == 5)]

dataset_712664 = dataset_712664.set_index('Movie_Id')

dataset_712664 = dataset_712664.join(df_title)['Name']

dataset_712664.head(10)
# Create a shallow copy for the movies dataset

user_712664 = df_title.copy()



user_712664 = user_712664.reset_index()



#To remove all the movies rated less often 

user_712664 = user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]

# getting 10K dataset

data = Dataset.load_from_df(dataset[['Cust_Id', 'Movie_Id', 'Rating']][:10000], reader)
#create a training set for svd

trainset = data.build_full_trainset()

svd.fit(trainset)



#Predict the ratings for user_712664

user_712664['Estimate_Score'] = user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est)



#Drop extra columns from the user_712664 data frame

user_712664 = user_712664.drop('Movie_Id', axis = 1)
# Sort predicted ratings for user_712664 in descending order

user_712664 = user_712664.sort_values('Estimate_Score', ascending=False)



#Print top 10 recommendations

(user_712664.head(10))