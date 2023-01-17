import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math

import re

from scipy.sparse import csr_matrix

from surprise import Reader,Dataset,SVD,evaluate
df1=pd.read_csv("../input/combined_data_1.txt", header=None, names=["Cust_Id", "Rating"], usecols=[0,1])

df1["Rating"]=df1["Rating"].astype(float)



print("Dataset 1 shape: {}".format(df1.shape))

print("Dataset Examples:")

print(df1.iloc[::5000000,:])

df1.head()
# Loding the remaining 3 data sets #



#df2 = pd.read_csv('../input/combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

#df3 = pd.read_csv('../input/combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

#df4 = pd.read_csv('../input/combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])





#df2['Rating'] = df2['Rating'].astype(float)

#df3['Rating'] = df3['Rating'].astype(float)

#df4['Rating'] = df4['Rating'].astype(float)



#print('Dataset 2 shape: {}'.format(df2.shape))

#print('Dataset 3 shape: {}'.format(df3.shape))

#print('Dataset 4 shape: {}'.format(df4.shape))



# Combining the data sets #

df = df1

#df=df.append(df2)

#df=df.append(df3)

#df=df.append(df4)



df.index=np.arange(0,len(df))

print("Full data set shape: {}".format(df.shape))

print("Dataset Examples :")

print(df.iloc[::5000000,:])

df.Rating.value_counts()
p = df.groupby('Rating')['Rating'].agg(['count'])



movie_count = df.isnull().sum()[1]



cust_count = df['Cust_Id'].nunique() - movie_count



rating_count = df['Cust_Id'].count() - movie_count



ax = p.plot(kind='barh',legend=False,figsize = (15,10))

plt.title('Total pool: {:,}Movies, {:,}Customers, {:,}Ratings'.format(movie_count,cust_count,rating_count),fontsize=20)

plt.axis('off')



for i in range(1,6):

    ax.text(p.iloc[i-1][0]/4,i-1, "Rating {}: {:.0f}%".format(i,p.iloc[i-1][0]*100/p.sum()[0]), color='white', weight='bold')
# Data Cleaning #
df_nan = pd.DataFrame(pd.isnull(df.Rating))

df_nan = df_nan[df_nan['Rating']==True]

df_nan = df_nan.reset_index()



movie_np = []

movie_id = 1



for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):

    temp = np.full((1,i-j-1), movie_id)

    movie_np = np.append(movie_np, temp)

    movie_id += 1

    

last_record = np.full((1,len(df) - df_nan.iloc[-1,0] - 1), movie_id)

movie_np = np.append(movie_np, last_record)



print('Movie numpy: {}'.format(movie_np))

print('Length: {}'.format(len(movie_np)))
# Removing the default movie_id rows #



df = df[pd.notnull(df['Rating'])]



df['Movie_Id'] = movie_np.astype(int)

df['Cust_Id'] = df['Cust_Id'].astype(int)

print('Dataset Examples: ')

print(df.iloc[::5000000, :])
# Eliminating movies with too less reviews and customers who give too less reviews #



f = ['count','mean']

df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)

df_movie_summary.index = df_movie_summary.index.map(int)

movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)

drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index



print('Movie minimum times of review: {}'.format(movie_benchmark))



df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)

df_cust_summary.index = df_cust_summary.index.map(int)

cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)

drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index



print("Customer minimum times of review: {}".format(cust_benchmark))

print("Original Shape : {}".format(df.shape))

df = df[~df['Movie_Id'].isin(drop_movie_list)]

df = df[~df['Cust_Id'].isin(drop_cust_list)]

print("After Trim Shape: {}".format(df.shape))

print("Data Examples: ")

print(df.iloc[::5000000, :])
# Putting the data into a matrix #



df_p = pd.pivot_table(df,values = 'Rating',index = 'Cust_Id', columns = 'Movie_Id')

print(df_p.shape)
# Loading the movie_mapping_file #



df_title = pd.read_csv('../input/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])

df_title.set_index('Movie_Id', inplace = True)

print(df_title.head(10))
# Recommend with Collaborative Filtering #



reader = Reader()



# Just acquire top 100k rows for faster run time #

data = Dataset.load_from_df(df[['Cust_Id','Movie_Id','Rating']][:100000], reader)

data.split(n_folds=3)



svd = SVD()

evaluate(svd, data, measures = ['RMSE', 'MAE'])
# What user 785314 liked in the past #



df_785314 = df[(df['Cust_Id'] == 785314) & (df['Rating'] == 5)]

df_785314 = df_785314.set_index('Movie_Id')

df_785314 = df_785314.join(df_title)['Name']

print(df_785314)
# Predicting which movies user_785314 would love to watch #



user_785314 = df_title.copy()

user_785314 = user_785314.reset_index()

user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]



# Getting full dataset

data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)



trainset = data.build_full_trainset()

svd.train(trainset)



user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)



user_785314 = user_785314.drop('Movie_Id', axis=1)



user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)

print(user_785314.head(10))
# Recommend with Pearsons' R Correlations



def recommend(movie_title, min_count):

    print("For movie ({})".format(movie_title))

    print("Top 10 movies recommended based on Pearsons' R Correlation : ")

    i = int(df_title.index[df_title['Name'] == movie_title][0])

    target = df_p[i]

    similar_to_target = df_p.corrwith(target)

    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])

    corr_target.dropna(inplace = True)

    corr_target = corr_target.sort_values('PearsonR', ascending = False)

    corr_target.index = corr_target.index.map(int)

    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR','Name','count','mean']]

    print(corr_target[corr_target['count'] > min_count][:10].to_string(index = False))
# Lets now try a recommendation for you if you like "What the #$*! Do We Know!?"
recommend("What the #$*! Do We Know!?", 0)
recommend("X2: X-Men United", 0)