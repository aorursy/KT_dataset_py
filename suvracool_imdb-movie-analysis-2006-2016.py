# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the csv file and creating a dataframe object

df = pd.read_csv('../input/IMDB-Movie-Data.csv')
df.describe()
##Removing all space and brackets in the column names

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df.describe()
##Dropping NaN columns from the below columns as i will be doing analysis on these columns

df.dropna(subset=['rating', 'director','year','director'],inplace=True)
##creating a sorted array for unique number of years

np1=df.year.unique()

list1=np.sort(np1)



count=[]

##Iterating through numpy array using nditer to fetch movies more than 300 million dollars revenue each year

for i in np.nditer(list1):

    j = len(df.loc[(df.revenue_millions>300) & (df.year==i)])

    count.append(j)

    

    

print(count)
index=np.arange(len(list1))



#Defining a function to plot a bar chart for number of movies having more than $300 million revenue each year.

def plot_bar_x():

    # this is for plotting purpose

    #index = np.arange(len(label))

    plt.bar(index,count)

    plt.xlabel('Year', fontsize=10)

    plt.ylabel('No of Movies', fontsize=10)

    plt.xticks(index, list1, fontsize=10, rotation=30)

    plt.title('Number of movies having revenue greater than 300 each year')

    plt.show()

    

plot_bar_x()
rating=[]

##Iterating through numpy array using nditer to fetch movies more than rating 8 each year

for i in np.nditer(list1):

    j = len(df.loc[(df.rating>8) & (df.year==i)])

    rating.append(j)

    

index=np.arange(len(list1))



#Defining a function to plot a bar chart for number of movies having more than $300 million revenue each year.

def plot_bar_x():

    # this is for plotting purpose

    #index = np.arange(len(label))

    plt.bar(index,rating,color='green')

    plt.xlabel('Year', fontsize=10)

    plt.ylabel('No of Movies', fontsize=10)

    plt.xticks(index, list1, fontsize=10, rotation=30)

    plt.title('Counting number of movies having rating greater than 8 each year')

    plt.show()

    

plot_bar_x()
## Fetching top 5 highest rated movie title and its director

df.nlargest(5,'rating')[['title','director','rating','revenue_millions']].set_index('title')
## Fetching top 5 highest revenue movie title and its director

df.nlargest(5,'revenue_millions')[['title','director','rating','revenue_millions']].set_index('title')
## Fetching top 5 lengthy movies title and its director

df.nlargest(5,'runtime_minutes')[['title','director','rating','runtime_minutes']].set_index('title')
##Top 10 directors based on IMDB rated movies who has directed more than 1 movie

##df.groupby('director').count()['rank'] >1 --> returns a series object where it tells whether the director has more than 1 movie 

##in the list. It is being converted to lost using tolist() method to use it as boolean indexer

df.loc[(df.groupby('director').count()['rank'] >1).tolist()].groupby('director').mean()[['rating']].nlargest(10,'rating')
##Average rating of movies yearwise

df.groupby('year').mean()[['rating']].nlargest(10,'rating')
##Violin plot of yearwise rating distribution using seaborn

df_yearwise_rating=df[['year','rating']]

ax = sns.violinplot(x="year", y="rating", data=df_yearwise_rating)
##Violin plot of yearwise revenue distribution using seaborn

df_yearwise_revenue=df[['year','revenue_millions']]

ax = sns.violinplot(x="year", y="revenue_millions", data=df_yearwise_revenue)