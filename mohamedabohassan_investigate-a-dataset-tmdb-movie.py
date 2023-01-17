#import packages 

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

% matplotlib inline
# Load data 

df=pd.read_csv("../input/tmdbmovies/tmdb-movies.csv")
df.head()
#last 10 rows

df.tail(10)
#shape of the dataframe

df.shape
df.info()
#check for null values

df.isnull().sum()
#name of columns

df.columns
# remove columns thats we don't need in the analysis 

df.drop(["id", 'imdb_id','cast', 'homepage','tagline', 'keywords', 'overview','original_title',

       'cast','director','production_companies', 'release_date',"vote_count","budget","revenue"], axis=1 , inplace= True )

df.head()
# the new shape od data frame

df.shape
#chech for duplicated 

sum(df.duplicated())
# the duplicated row

df[df.duplicated()]
#remove the duplicated row

df.drop_duplicates(inplace=True)
#chech for duplicated again

sum(df.duplicated())
#check of null values

df.isnull().sum()
#remove rows with null values

df.dropna(inplace=True)
##check of null values

df.isnull().sum()
#check if dataframe has zero elemens

df.eq(0).any().any()
# creates a boolean dataframe which is True where df is nonzero

df != 0
# Remove rows with zero values.

df = df[(df != 0).all(1)]

df.head()
#check if dataframe has zero elemens

df.eq(0).any().any()
#the new data shape

df.shape
data = df.join(df.genres.str.strip('|').str.split('|',expand=True).stack().reset_index(level=1,drop=True).rename('genre')).reset_index(drop=True)



data.head(8)
#check on clean data before exploring

df.isnull().sum()
#some statistics

data.describe()
# the mean adjusted budget by genre

mean_genre_budget = data.groupby('genre').budget_adj.mean()



# the mean popularity by genre

mean_genre_popularity = data.groupby('genre').popularity.mean()



# the mean runtime by genre

mean_genre_runtime = data.groupby('genre').runtime.mean()



# the mean release year by genre

mean_genre_release_year = data.groupby('genre').release_year.mean()



# the mean vote average by genre

mean_genre_vote = data.groupby('genre').vote_average.mean()



#  the mean adjusted revenue by genre

mean_genre_revenue = data.groupby('genre').revenue_adj.mean()

mean_genre_data = pd.concat([ mean_genre_budget,mean_genre_popularity, mean_genre_revenue, mean_genre_runtime, mean_genre_release_year, mean_genre_vote],axis=1)

mean_genre_data
#make the genre as column at mean_genre_data

all_mean = mean_genre_data.reset_index()

all_mean
# Distribution of Popularity vs genre

mean_genre_data["popularity"].hist(figsize=(8,5));

#set the title of figure

plt.title('Distribution of Popularity vs genre')

#set the xlabel and y label of the figure

plt.xlabel('Popularity')

plt.ylabel('Frequency of Occurence')
#sorting the highest popularity by genre

mean_genre_data.sort_values('popularity', ascending=False).popularity

# visualize the relations between the genres and popularity



x=all_mean["genre"]

y=all_mean["popularity"]

fig,ax = plt.subplots(figsize=(10,8))

ax.bar(x, y)

plt.xticks(rotation=90)

plt.title("popular genre",fontsize=12)

plt.xlabel('genre Of Movies',fontsize=12)

plt.ylabel("popularity",fontsize= 12)

# visualize the relations between the genres and budget

x=all_mean["genre"]

y=all_mean["budget_adj"]

fig,ax = plt.subplots(figsize=(10,8))

ax.bar(x, y)

plt.xticks(rotation=90)

#setup the title of the figure

plt.title("Highest budget vs genres",fontsize=12)

#setup the xlabel and ylabel of the figure

plt.xlabel('genre Of Movies',fontsize=12)

plt.ylabel("budget",fontsize= 12)
#sorting the highest budget by genre



mean_genre_data.sort_values('budget_adj', ascending=False).budget_adj

#plot the relations between the genres and budget

x=all_mean["genre"]

y=all_mean["revenue_adj"]

fig,ax = plt.subplots(figsize=(10,8))

ax.bar(x, y)

plt.xticks(rotation=90)

#setup the title of the figure

plt.title("Highest revenue vs genres",fontsize=12)

#setup the xlabel and ylabel of the figure

plt.xlabel('genre Of Movies',fontsize=12)

plt.ylabel("revenue",fontsize= 12)
#sorting the highest revenue by genre

mean_genre_data.sort_values('revenue_adj', ascending=False).revenue_adj

#avarge run time 

data["runtime"].mean()
#make a Group of mean runtime by release_year

#i used here the clean datafram 'data' to make the group

mean_rel_run= data.groupby('release_year').runtime.mean()

mean_rel_run
#plot the relations between runtime and release year between years (1960 and 2015)

mean_rel_run.plot(xticks = np.arange(1960,2016,5),figsize=(10,8));

#setup the title of the figure

plt.title("Runtime Vs  year",fontsize = 14)

#setup the x-label and y-label of the plot.

plt.xlabel(' year',fontsize = 13)

plt.ylabel('Runtime',fontsize = 13)
# the relations between the genres and popularity

x=all_mean["genre"]

y=all_mean["vote_average"]

fig,ax = plt.subplots(figsize=(10,8))

ax.bar(x, y)

plt.xticks(rotation=90)

plt.title("Highest vote of the genre",fontsize=12)

plt.xlabel('genre Of Movies',fontsize=12)

plt.ylabel("vote_average",fontsize= 12)
#sorting the highest vote average by genre



mean_genre_data.sort_values('vote_average', ascending=False).vote_average
