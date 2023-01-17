#importing the necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#loading the dataset

data=pd.read_csv("../input/tmdb_movies_data.csv")

data.info()
#getting the first few rows of the dataset

data.head()
#printing the last 5 rows

data.tail(5)
#summary of dataset

data.describe()
#filling na values with 0

data.fillna(0)

print()
#total number of rows and columns initially in the dataset

print(data.shape)
#we need to delete the initially duplicated present rows

#for that, first we will count the total numbers of dupicated rows

sum(data.duplicated())

#now we will remove the duplicated rows

data.drop_duplicates(inplace=True)

print(data.shape)
#now dropping the columns which will not be helpful in the data analysis

data.drop(['imdb_id','overview','cast','homepage','budget_adj','revenue_adj','keywords','tagline'],axis=1,inplace=True)

print(data.shape)
#the release date column is in the string format. we need to change it to datetime format.

data['release date']=pd.to_datetime(data['release_date'])

data['release date'].head()
#making a new column

data['profit']=data['revenue']-data['budget']
#Finding the higest and the lowest grossing movies

#the idxmax() function gives the index of the maximum value  

high_pr=data['profit'].idxmax()

#the idxmin() function gives the index of the minimm value

low_pr=data['profit'].idxmin()

print('The highest Grossing movie of all time is: ',data['original_title'][high_pr],'with Profit: $',data['profit'][high_pr])

print('The lowest Grossing movie of all time is: ',data['original_title'][low_pr],'with Profit: $',data['profit'][low_pr])
#some of the columns have 0 as value. So we change to replace them with NaN so that it dont affect the result

data['budget']=data['budget'].replace(0,np.NAN)

high_bud=data['budget'].idxmax()

low_bud=data['budget'].idxmin()

print('The movie with highest budget is: ',data['original_title'][high_bud])

print('The movie with lowest budget is: ',data['original_title'][low_bud])
data['revenue']=data['revenue'].replace(0,np.NAN)

high_rev=data['revenue'].idxmax()

low_rev=data['revenue'].idxmin()

print('The movie with highest budget is: ',data['original_title'][high_rev])

print('The movie with lowest budget is: ',data['original_title'][low_rev])
long_rt=data['runtime'].idxmax()

short_rt=data['runtime'].idxmin()

print('The movie with the longest running time is:',data['original_title'][long_rt])

print('The movie with the shortest running time is:',data['original_title'][short_rt])
high_vote=data['vote_average'].idxmax()

low_vote=data['vote_average'].idxmin()

print('The movie with the highest votes is:',data['original_title'][high_vote],'with average',data['vote_average'][high_vote])

print('The movie with the lowest votes is:',data['original_title'][low_vote],'with average',data['vote_average'][low_vote])
def count_genre(x):

    data_plot=data[x].str.cat(sep='|')

    dat=pd.Series(data_plot.split('|'))

    info=dat.value_counts(ascending=False)

    return info

sum_gen=count_genre('genres')

sum_gen.plot(kind='bar',figsize=(15,7),fontsize=12)

plt.xticks(rotation=60)

plt.title("Genre with Highest Release",fontsize=15)

plt.xlabel("Genres",fontsize=14)

plt.ylabel("Movies",fontsize=14)
info=pd.DataFrame(data['profit'].sort_values(ascending=False))

info['original_title']=data['original_title']

dat=list(map(str,(info['original_title'])))

x=list(dat[:10])

y=list(info['profit'][:10])

p=sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(10,5)})

plt.title("Top 10 Profitable Movies",fontsize=15)

plt.xlabel("Profit",fontsize=12)

sns.set_style('darkgrid')
info=pd.DataFrame(data['budget'].sort_values(ascending=False))

info['original_title']=data['original_title']

dat=list(map(str,(info['original_title'])))

x=list(dat[:10])

y=list(info['budget'][:10])

p=sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(10,5)})

plt.title("Top 10 Highest Budget Movies",fontsize=15)

plt.xlabel("Budget(in Billions)",fontsize=12)
info=pd.DataFrame(data['vote_average'].sort_values(ascending=False))

info['original_title']=data['original_title']

dat=list(map(str,(info['original_title'])))

x=dat[:10]

y=info['vote_average'][:10]

p=sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(10,5)})

plt.title("Top 10 highest voted movies",fontsize=15)

plt.xlabel("Average Vote",fontsize=12)
info=pd.DataFrame(data['runtime'].sort_values(ascending=False))

info['original_title']=data['original_title']

dat=list(map(str,(info['original_title'])))

x=list(dat[:10])

y=list(info['runtime'][:10])

p=sns.pointplot(x=y,y=x)

sns.set(rc={'figure.figsize':(10,5)})

plt.title("Top 10 longest running movies",fontsize=15)

plt.xlabel("Run Time(in minutes)",fontsize=12)
data.groupby('release_year')['profit'].mean().plot()

plt.title('Year vs Average Profit',fontsize=15)

plt.xlabel('Release Year',fontsize=12)

plt.ylabel('Average Profit',fontsize=12)

plt.yticks(rotation=90)
data.groupby('release_year')['runtime'].mean().plot()

plt.title('Average Runtime vs Release Years',fontsize=15)

plt.xlabel('Release Year',fontsize=12)

plt.ylabel('Average Runtime (in minutes)',fontsize=12,rotation=90)
data.groupby('runtime')['popularity'].mean().plot(xticks=np.arange(0,1000,100))

plt.title('Running Time vs Popluarity',fontsize=15)

plt.xlabel('Running Time (in minutes)',fontsize=12)
dc=data.corr()

p=sns.regplot(x='release_year',y='vote_average',data=data)

plt.title("Average Votes vs Release Years",fontsize=15)

plt.xlabel('Release Year',fontsize=12)

plt.ylabel('Average Votes',fontsize=12)

print('The correlation between Average Votes and Release Years is:',dc.loc['vote_average']['release_year'])
p=sns.regplot(x='budget',y='profit',data=data)

#data.columns