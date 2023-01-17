import pandas as pd

import pickle

import numpy as np

from fastai.collab import *

from pprint import pprint

import matplotlib.pyplot as plt

import umap

from scipy import stats

%matplotlib inline
# the original csv from https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/master/

# The column ID is used in API calls to retrieve the game reviews

games = pd.read_csv('../input/2019-05-02.csv')

games.describe()

games.sort_values('Users rated',ascending=False,inplace=True)

games.rename(index=str, columns={"Bayes average": "Geekscore",'Name':'name'}, inplace=True)

games[:10]
reviews = pd.read_csv('../input/bgg-13m-reviews.csv',index_col=0)

print(len(reviews))

reviews.head()
reviews['rating'].hist(bins=10)

plt.xlabel('rating of review')

plt.ylabel('number of reviews')

plt.show()
games_by_all_users = reviews.groupby('name')['rating'].agg(['mean','count']).sort_values('mean',ascending=False)

games_by_all_users['rank']=games_by_all_users.reset_index().index+1

print(len(games_by_all_users))



games_by_all_users = games_by_all_users.merge(games[['name','Geekscore']],how='left',left_on=['name'], right_on=['name'])

games_by_all_users.head()
x = games_by_all_users['rank']

y = games_by_all_users['mean']

plt.figure(num=None, figsize=(7, 3), facecolor='w', edgecolor='k')

plt.scatter(x, y,s=0.5)

plt.xlabel('sorted games by average rating')

plt.ylabel('average rating')

plt.show()  # or plt.savefig("name.png")
games_by_all_users[['mean','Geekscore']].hist(bins=10)

plt.xlabel('averge rating of game')

plt.ylabel('number of games')

plt.show()
x = games_by_all_users['count']

y = games_by_all_users['Geekscore']

y2 = games_by_all_users['mean']



df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data

data_cut = pd.cut(df.X,bins=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000,10000,100000])           #we cut the data following the bins

grp = df.groupby(by = data_cut)        #we group the data by the cut

ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin





df2 = pd.DataFrame({'X' : x, 'Y' : y2})  #we build a dataframe from the data

data_cut = pd.cut(df2.X,bins=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000,10000,100000])           #we cut the data following the bins

grp = df2.groupby(by = data_cut)        #we group the data by the cut

ret2 = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin



#plotting

plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')

plt.xscale('log')

plt.scatter(df2.X,df2.Y,alpha=.5,s=0.5)

plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)

plt.plot(ret2.X,ret2.Y,'r--',lw=4,alpha=0.5)

plt.xlabel('number of reviews')

plt.ylabel('average rating of game')

plt.show()
reviews_by_user_count = reviews.groupby('user')['rating'].agg(['mean','count']).sort_values('count',ascending=False).reset_index()

print(len(reviews_by_user_count))

reviews_by_user_count.head()
reviews_by_user_count['count'].hist(log=True)

plt.xlabel('number of users')

plt.ylabel('number of reviews written')
# select users that reviewed more than cutoff games

cutoff = 500

active_users = reviews_by_user_count[reviews_by_user_count['count']>cutoff]

active_users = active_users['user']

reviews_by_active_users = reviews[reviews['user'].isin(active_users)]

print(len(reviews_by_active_users))

reviews_by_active_users.head()
count_user, count_review = reviews[['user','name']].nunique()

print('density',len(reviews)/(count_user*count_review))

count_user, count_review = reviews_by_active_users[['user','name']].nunique()

print('density', len(reviews_by_active_users)/(count_user*count_review))
games_rated_by_active_users = reviews_by_active_users.groupby('name')['rating'].agg(['mean','count']).sort_values('mean',ascending=False)

games_rated_by_active_users['rank']=games_rated_by_active_users.reset_index().index+1



print('{} users original, with {} reviews'.format(reviews['user'].nunique(),len(reviews)))

print('{} users left({}% of the userbase), with {} reviews (this is {} of all reviews)'.format(len(active_users),len(active_users)/reviews['user'].nunique(),len(reviews_by_active_users),len(reviews_by_active_users)/len(reviews)))
games_rated_by_active_users['mean'].hist()

games_by_all_users['mean'].hist()
x = reviews_by_user_count['count']

y = reviews_by_user_count['mean']

plt.xscale('log')



plt.hist2d(x, y, bins=[np.logspace(np.log10(30),np.log10(1000),40),np.linspace(5,10,num=40)], cmap=plt.cm.jet)

plt.colorbar()

plt.xlabel('Number of reviews written')

plt.ylabel('Average score of user')

plt.show()
merge = games_rated_by_active_users[['mean','count']].merge(games_by_all_users[['name','mean','count', 'Geekscore']],how='outer',left_index=True, right_on=['name'],suffixes=('active','all'),indicator=True)



merge['delta_active_all']=merge['meanactive']-merge['meanall']

merge['proportion_active']=merge['countactive']/merge['countall']

merge.sort_values('countall',ascending=False)[:5]
merge['delta_active_all'].median()
corr = merge.corr()

corr.style.background_gradient(cmap='coolwarm')
y = merge['meanall']

x = merge['proportion_active']



df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data

data_cut = pd.cut(df.X,bins=np.linspace(0,1,num=10))   

grp = df.groupby(by = data_cut)        #we group the data by the cut

ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin



#plotting

plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')

plt.scatter(df.X,df.Y,alpha=.5,s=0.5)

plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)

plt.xlabel('proportion of reviews given by active users')

plt.ylabel('average rating of game')

plt.show()
x = merge['countall']

y = merge['proportion_active']



df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data

data_cut = pd.cut(df.X,bins=np.logspace(0,5,num=30))           #we cut the data following the bins

grp = df.groupby(by = data_cut)        #we group the data by the cut

ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin



#plotting

plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')

plt.xscale('log')

plt.scatter(df.X,df.Y,alpha=.5,s=0.5)

plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)

plt.xlabel('number of reviews for a game')

plt.ylabel('proportion of reviews given by active users')

plt.show()
merged_reviews = reviews.merge(reviews_by_user_count,how='left',on='user',suffixes=('','user'),indicator=True)

games_test = merged_reviews.groupby('name')[['rating','count']].agg(['mean','median','count']).sort_values(('count', 'count'),ascending=False)

games_test[:5]
corr = games_test.corr()

corr.style.background_gradient(cmap='coolwarm')
games_test['count','median'].hist(bins=50)
games_test = games_test.sort_values(('count', 'median'),ascending=True)

games_test[games_test['rating', 'count']>0][:30]
y = games_test[games_test['rating', 'count']>0]['rating', 'mean']

x = games_test[games_test['rating', 'count']>0]['count', 'median']



df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data

data_cut = pd.cut(df.X,bins=np.logspace(0,3,num=40))   

grp = df.groupby(by = data_cut)        #we group the data by the cut

ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin



#plotting

plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')

plt.xscale('log')

plt.scatter(df.X,df.Y,alpha=.5,s=0.5)

plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)

plt.xlabel("median number of reviews of a game's userbase (the higher the more active the userbase of a game)")

plt.ylabel('average score for a game')

plt.show()
x = games_test['rating', 'count']

y = games_test['count', 'median']





df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data

data_cut = pd.cut(df.X,bins=np.logspace(0,5,num=30))   

grp = df.groupby(by = data_cut)        #we group the data by the cut

ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin



#plotting

plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')

plt.xscale('log')

plt.scatter(df.X,df.Y,alpha=.5,s=0.5)

plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)

plt.xlabel('number of reviews per game')

plt.ylabel("median number of reviews of a game's userbase")

plt.show()