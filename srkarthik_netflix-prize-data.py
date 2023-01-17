#importing required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graph plots to understand data
# read the first data file as csv using pandas

df1 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header = None, names = ['Customer_Id', 'Rating'], usecols = [0,1])

# cast rating column to float type

df1['Rating'] = df1['Rating'].astype(float)



print('Dataset 1 shape: {}'.format(df1.shape))

print('-Dataset examples-')

print(df1.iloc[::5000000, :])
# read other data files as csv and append data together



# commented as my machine lags even for 1 file

#df2 = pd.read_csv('../input/netflix-prize-data/combined_data_2.txt', header = None, names = ['Customer_Id', 'Rating'], usecols = [0,1])

#df3 = pd.read_csv('../input/netflix-prize-data/combined_data_3.txt', header = None, names = ['Customer_Id', 'Rating'], usecols = [0,1])

#df4 = pd.read_csv('../input/netflix-prize-data/combined_data_4.txt', header = None, names = ['Customer_Id', 'Rating'], usecols = [0,1])





#df2['Rating'] = df2['Rating'].astype(float)

#df3['Rating'] = df3['Rating'].astype(float)

#df4['Rating'] = df4['Rating'].astype(float)



#print('Dataset 2 shape: {}'.format(df2.shape))

#print('Dataset 3 shape: {}'.format(df3.shape))

#print('Dataset 4 shape: {}'.format(df4.shape))
# combine all data

df = df1

#df = df1.append(df2)

#df = df.append(df3)

#df = df.append(df4)



df.index = np.arange(0,len(df))

print('Full dataset shape: {}'.format(df.shape))

print('-Dataset examples-')

print(df.iloc[::5000000, :])
# group by rating column with aggregation function count

p = df.groupby('Rating')['Rating'].agg(['count'])

print(p.head())

# get movie count

movie_count = df.isnull().sum()[1]



# get customer count

cust_count = df['Customer_Id'].nunique() - movie_count



# get rating count

rating_count = df['Customer_Id'].count() - movie_count



ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))

plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
# create a data frame with index of movie id rows

df_nan = pd.DataFrame(pd.isnull(df.Rating))

df_nan = df_nan[df_nan['Rating'] == True]

df_nan = df_nan.reset_index()

print(df_nan.head())
# scratch line just for understanding creating the required iteration conditions

temp = np.full((1,5), 1)

print(temp)

movie_np = []

movie_id =1 

for i,j in zip(df_nan['index'][1:],df_nan['index'][:2]):

    print(i)

    print(j)

    # numpy approach

    temp = np.full((1,i-j-1), movie_id)

    print(temp)

    movie_np = np.append(movie_np, temp)

    print(len(temp[0]))

    movie_id += 1
# create a new array and fill movie id as new column

movie_np = []

movie_id = 1



for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):

    # numpy approach

    temp = np.full((1,i-j-1), movie_id)

    movie_np = np.append(movie_np, temp)

    movie_id += 1



# Account for last record and corresponding length

# numpy approach

last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)

movie_np = np.append(movie_np, last_record)



print('Movie numpy: {}'.format(movie_np))

print('Length: {}'.format(len(movie_np)))
# remove those Movie ID rows

df = df[pd.notnull(df['Rating'])]



df['Movie_Id'] = movie_np.astype(int)

df['Customer_Id'] = df['Customer_Id'].astype(int)

print('-Dataset examples-')

print(df.iloc[::5000000, :])

print(df.head())
"""

we can eliminate movies that have very less number of ratings which is obviously not a good movie and 

also those customers who vote very less number of times as these accounts can be considered dormant accounts 

"""

f = ['count','mean']

#  setting a benchmark by creating list of  movie id whose rating quantile is 0.7

df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)

print(df_movie_summary.head())

df_movie_summary.index = df_movie_summary.index.map(int)

print(df_movie_summary.index.map(int))

movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)

drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print(drop_movie_list)

print('Movie minimum times of review: {}'.format(movie_benchmark))

#  setting a benchmark by creating list of  customer id whose rating quantile is 0.7

df_cust_summary = df.groupby('Customer_Id')['Rating'].agg(f)

print(df_cust_summary.head())

df_cust_summary.index = df_cust_summary.index.map(int)

print(df_cust_summary.index.map(int))

cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)

drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

print(drop_cust_list)

print('Customer minimum times of review: {}'.format(cust_benchmark))
# remove the lower reviews from actual data

print('Original Shape: {}'.format(df.shape))

df = df[~df['Movie_Id'].isin(drop_movie_list)]

df = df[~df['Customer_Id'].isin(drop_cust_list)]

print('After Trim Shape: {}'.format(df.shape))

print('-Data Examples-')

print(df.iloc[::5000000, :])

print(df.head())
# read the movie list csv file

df_title = pd.read_csv('../input/netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])

df_title.set_index('Movie_Id', inplace = True)

print (df_title.head(10))

print(df_title.shape)
# piovt the data since most algorithms need data as sparse matrix, this will be used in Pearson's R correlation method, and SVD

# pivoting only first 100000 rows due to system memory constraints

df_p = pd.pivot_table(df[:100000],values='Rating',index='Customer_Id',columns='Movie_Id')



print(df_p.shape)

print(df_p.head())
"""

to predict the top 10 movies realted to a particular movie so that we can recommend movies based on movie that has been 

watched by the customer. for this we are using pearson's R correlation method



in pearson's R  correlation method is to find the streangth of association between any two variables

"""

def recommend(movie_title, min_count):

    # get movie name and minimum count of rating

    print("For movie ({})".format(movie_title))

    print("- Top 10 movies recommended based on Pearson's R correlation - ")

    # get the movie id from movie list data frame and compare it to the pivot table that we have constructed in previous step

    i = int(df_title.index[df_title['Name'] == movie_title][0])

    # get the row from pivot table for the watched movie

    target = df_p[i]

    # to find the similar movies correlate the target dataframe with the pivoted data frame

    similar_to_target = df_p.corrwith(target)

    print(similar_to_target.head())

    # create correlated suggestion list as new data from with pearsonR as the column name for the correlation value

    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])

    # remove all null values from the data frame

    corr_target.dropna(inplace = True)

    # sort the data frame in descending order to get the most recommended movie to the top of the list 

    corr_target = corr_target.sort_values('PearsonR', ascending = False)

    corr_target.index = corr_target.index.map(int)

    """

    join the correlation target data frame wiht movie list and benchmarked movie summary dataframe to get movie name and 

    count and mean of rating for each of them

    """

    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]

    # filter the top ten rows with count greater than minimum count to get hte top 10 recommended movie list

    print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))
# invoke recommend function to get the top 10 recommended movie list

recommend("Character", 0)
# surprise package is not available by default in anaconda so to install execute below 

#conda install -c conda-forge scikit-surprise
# Import required libraries

import math

import re

import matplotlib.pyplot as plt



from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate
# SVD requires dataset as reader object

# Load Reader library

reader = Reader()



# get just top 100K rows for faster run time

data = Dataset.load_from_df(df[['Customer_Id', 'Movie_Id', 'Rating']][:100000], reader)



# Use the SVD algorithm.

svd = SVD()



# Compute the RMSE of the SVD algorithm

"""

most ML algorithms require RMSE and MAE value which are Root Mean Square Error and Mean Absolute Error. In simple terms the ML 

algorithms need effective error value and actual error value

"""

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
df.head()
#To find all the movies rated as 5 stars by user with userId = 712664

dataset_712664 = df[(df['Customer_Id'] == 712664) & (df['Rating'] == 5)]

dataset_712664 = dataset_712664.set_index('Movie_Id')

dataset_712664 = dataset_712664.join(df_title)['Name']

dataset_712664.head(10)
#Train an SVD to predict ratings for user with userId = 1

# Create a shallow copy for the movies dataset

user_712664 = df_title.copy()



user_712664 = user_712664.reset_index()



#To remove all the movies rated less often 

user_712664 = user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]


# getting full dataset

data = Dataset.load_from_df(df[['Customer_Id', 'Movie_Id', 'Rating']], reader)



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

print(user_712664.head(10))