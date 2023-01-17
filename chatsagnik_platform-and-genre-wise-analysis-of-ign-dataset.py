import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ign = pd.read_csv('../input/ign.csv')

ign.head(5)
ign.columns
ign.drop(['Unnamed: 0','score_phrase','title','url','editors_choice','release_day','release_month','release_year'],axis = 1,inplace = True)

ign.columns
ign.head(3)
## Now we would like to analyse the score of games by the remaining columns

platform_stats = ign.groupby('platform').score.agg(['count','mean'])

platform_stats = platform_stats.reset_index()

platform_stats.shape
## Throwing down basic plots - platform vs No of games, and platform vs mean score

## However there are 59 different types of platforms, so we must sort and display only the top 10 for each plot

pls_count = platform_stats.sort_values(['count'],ascending = False).head(10)

pls_score = platform_stats.sort_values(['mean'],ascending = False).head(10)
## Time to plot Platform vs Number of Games released



plt.rcParams['figure.figsize']=(20,10)

ax = sns.barplot(x = 'platform', y = 'count',data = pls_count)



plt.xlabel("Major Platforms")

plt.ylabel("Number of Games released for platform")



plt.show()
## Time to plot Platform vs Average Score of Game



plt.rcParams['figure.figsize']=(20,10)

ax = sns.barplot(x = 'platform', y = 'mean',data = pls_score)



plt.xlabel("Platforms with highest scores")

plt.ylabel("Average Score of Games released for platform")



plt.show()
## Performing same analysis for genre

genre_stats = ign.groupby('genre').score.agg(['count','mean'])

genre_stats = genre_stats.reset_index()

genre_stats.shape
gs_count = genre_stats.sort_values(['count'],ascending = False).head(10)

gs_score = genre_stats.sort_values(['mean'],ascending = False).head(10)
## Time to plot Genre vs Number of Games released



plt.rcParams['figure.figsize']=(20,10)

ax = sns.barplot(x = 'genre', y = 'count',data = gs_count)



plt.xlabel("Major Genres")

plt.ylabel("Number of Games released for each Genre")



plt.show()
## Time to plot Genre vs Average Score of Game



plt.rcParams['figure.figsize']=(20,10)

ax = sns.barplot(x = 'genre', y = 'mean',data = gs_score)



plt.xlabel("Genres with highest scores")

plt.ylabel("Average Score of Games released for each Genre")



plt.show()