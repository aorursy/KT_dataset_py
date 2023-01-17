import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df_movies = pd.read_csv('C:/Users/kumarijy/Documents/Learning/DAND/data_analysis_project/tmdb-movies.csv')
df_movies.head()

#print("Movies DF:\n\n{}\n".format(df_movies.head()))
#df_movies.drop('keywords',axis = 1,inplace = True)

df_movies.drop('imdb_id',axis = 1,inplace = True)

df_movies.drop('homepage',axis = 1,inplace = True)

df_movies.drop('overview',axis = 1,inplace = True)

df_movies.drop('tagline',axis = 1,inplace = True)

df_movies.drop('vote_count', axis = 1, inplace = True)
df_movies = pd.read_csv('tmdb-movies.csv')

sum(df_movies.duplicated())
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%m/%d/%y')

df_movies['release_month'] = df_movies['release_date'].dt.month
df_movies['profit'] = df_movies['revenue'] - df_movies['budget']
df_movies.info()
df_top_budget = df_movies.nlargest(20,'budget')

df_top_budget.loc[:,['id','budget','revenue','original_title','director','vote_average','genres']]
df_top_revenue = df_movies.nlargest(20,'revenue')

df_top_revenue.loc[:,['id','budget','revenue','original_title','director','vote_average','genres']]
df_movies['release_month'].value_counts()
sns.distplot(df_movies['release_month'])
g = sns.regplot(x = 'budget', y='revenue' , data = df_movies)

# remove the top and right line in graph

sns.despine()

# Set the size of the graph from here

g.figure.set_size_inches(7,5)

# Set the Title of the graph from here

g.axes.set_title('Budget vs. Revenue', fontsize=20,color="b",alpha=0.8)

# Set the x & y label of the graph from here

g.set_xlabel("Budget",size = 20,color="c",alpha=0.8)

g.set_ylabel("Revenue",size = 20,color="c",alpha=0.8)
df = df_movies.query('budget > 0 & revenue > 0')
df.shape


f, ax = plt.subplots(figsize=(7, 7))

ax.set( yscale="log")

p = sns.regplot('vote_average','budget', data =df, ax = ax, scatter = True)

p.set(ylabel='log(budget)')

p.axes.set_title('Log(Budget) vs User Rating', fontsize=20,color="c",alpha=0.8)
#Array with the column names for what we want to compare the revenue to

revenue_comparisons = ['budget', 'runtime', 'vote_average', 'popularity','release_month']

for comparison in revenue_comparisons:

    sns.jointplot(y='revenue', x=comparison, data=df_movies, color='b', size=5, space=0, kind='reg')

    #p.axes.set_title('y vs x', fontsize=20,color="c",alpha=0.8)
ax = sns.distplot(df_movies['release_year'])

ax.set_title("Growth of movies production with years", color = 'c')

col = df_movies['genres']
col2 = []

for s in col:

    #print(s)

    try:

        x = s.split('|')

    except:

        x = ['No']

    col2.append(x)
l1 =[]

for s in col2:

    #print(type(s))

    l1 = sum([l1,s],[])



gener = set(l1)
gener = list(gener)

gener.remove('No')
gener
# Here we have used df dataset which will have data for all the records having revenue and budget 

#both greater than zero

df.columns
for g in gener:

    df1 = df['genres'].str.contains(g).fillna(False)

    #print('The total number of movies with ',g,'=',len(df[df1]))

    f, ax = plt.subplots(figsize=(25, 5))

    sns.countplot(x = 'release_year', data=df[df1], palette="Greens_d")

    plt.title(g)

    compare_movies_rating = ['budget']

    for compare in compare_movies_rating:

        sns.jointplot(y ='profit', x=compare, data=df[df1], alpha=0.7, color='b', size=5)

        plt.title(g)
# Here we used df dataset so that the records having valid revenue and budget should be considered

df_profit = df.nlargest(20,'profit')



df_profit.genres.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('TOP 20 GENRE IN MOVIE DATASET ')