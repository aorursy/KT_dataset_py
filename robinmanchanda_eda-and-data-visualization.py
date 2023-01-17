import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
df_rating=pd.read_csv('../input/movielens-20m-dataset/rating.csv')

df_gscores=pd.read_csv('../input/movielens-20m-dataset/genome_scores.csv')

df_gtag=pd.read_csv('../input/movielens-20m-dataset/genome_tags.csv')

df_link=pd.read_csv('../input/movielens-20m-dataset/link.csv')

df_movie=pd.read_csv('../input/movielens-20m-dataset/movie.csv')

df_tag=pd.read_csv('../input/movielens-20m-dataset/tag.csv')
df_movie.head()
df_movie.isnull().any()
print('The total number of movies in the dataset : {}'.format(len(df_movie['movieId'].unique())))
print('The total number of unique movies(title) in the dataset : {}'.format(len(df_movie['title'].unique())))
print('The total number of dublicate movies( Two movieIds with same title 16*2=32) in the dataset : {}'.format(len(df_movie[df_movie.duplicated(subset = 'title', keep = 'first')])))

df_tmp=df_movie[df_movie.duplicated(subset = 'title', keep = False)]

df_tmp
#  Loop to find dublicate movies with same and different genres list.

lis1=[]

lis2=[]

for j in range(32):

    x=0

    for i in range(31):

        if df_tmp.iloc[j,1]== df_tmp.iloc[i+1,1]:

            if df_tmp.iloc[j,2]== df_tmp.iloc[i+1,2]:

                x+=1

                if x>1:

#                 print(j, i)

                    lis1.append(df_tmp.iloc[j,1])

                else:

                    lis2.append(df_tmp.iloc[j,1])

                    

print('List of dublicate movies with same genres {}'.format(list(set(lis1))))

print('')

print('List of dublicate movies with different genres {}'.format(list(set(lis2)-set(lis1))))



            
# Here we have selected the first occurence of movie, as it has almost all the genres of the second title. 

df_movie.drop_duplicates(subset='title', inplace = True, keep= 'first')
df_movie[df_movie.duplicated(subset = 'title', keep = False)]

df_rating
plt.hist(df_rating['rating'])
print('The number of movies which are rated by users : {}'.format(len(df_rating['movieId'].unique())))
d=df_rating.groupby('movieId').mean()['rating']

df=pd.DataFrame(d, columns=['rating']).reset_index()

df.head()
sns.boxplot(df['rating'], orient='v')
print('Max rating : {}  Min rating : {}'.format(df['rating'].max(), df['rating'].min()))
df2=pd.merge(df_movie, df, on='movieId', how='left' )

df2['Year']=df2['title'].str.extract('.*\((.*)\).*')

df2['Year'].replace(['Das Millionenspiel','Bicicleta, cullera, poma','2007-','2009– ', '1975-1979', '1983)'], value=[np.nan, np.nan, '2007', '2009', '1975', '1983' ], inplace=True)
sns.boxplot(df2['Year'].dropna().astype('int64'), orient='v')
sns.distplot(df2['Year'].dropna().astype('int64'))
year=df2['Year'].dropna().astype('int64', copy= True).sort_values().unique().tolist()

year=list(map(str, year))
df7=df2[['Year', 'movieId']].groupby('Year').count()

df7.rename(mapper={'movieId': 'movies/year'}, inplace= True, axis=1)

df7.head()
df7.plot.bar(figsize=(20,15))

plt.ylabel('# movies', fontsize=12)

plt.title('Number of movies released in a particular year', fontsize=16)

plt.legend('')

plt.figure(figsize=(20,10))

plt.ylabel('Avg. Rating')

plt.xticks(rotation='vertical')

sns.lineplot(year, list(df2.groupby('Year')['rating'].mean().to_numpy()))

plt.title('Relation between Avg. rating and Year', fontsize=16)
# Loop to find Percentage of movies which are highly rated(>=4) for all years.



rating_lis=[]

for i in range(len(year)):

#     print(year[i])

    le=len(df2[(df2['rating']>=4) & (df2['Year']== year[i])])

    if le>0:

#         print(le/(df7.loc[year[i]])

        rating_lis.append((le/(df7.loc[year[i]]['movies/year']))*100)

    else:

        rating_lis.append(0) 

        

rating_lis[:5]
plt.figure(figsize=(20,10))

plt.plot(year,rating_lis)

plt.xticks(rotation='vertical')

plt.ylabel("Percentage of movies which are highly rated(>=4)", fontsize=12)

plt.title('Percentage of movies which are highly rated(>=4) W.R.T Year', fontsize=16)
df2[df2['Year']=='1901']
print('# Movies for which rating is not available:{}'.format(df2['rating'].isnull().sum()))
df2[df2['rating'].isnull()]['title'].head()
df_gscores.head()
# Code to extract higly relevant tagId 

df3=df_gscores[df_gscores.groupby('movieId')['relevance'].transform(max)==df_gscores['relevance']]

df3
df3[df3.duplicated(subset='movieId', keep= 'first' )]
df3=df3.drop_duplicates(subset='movieId')
df4=pd.merge(df3, df_gtag, how='left', on='tagId')

df4
df5=pd.merge(df2, df4, how='left', on='movieId')

df5.rename(mapper={'tag':'High_relevance_tag', 'tagId':'High_relevance_tagId'}, axis=1, inplace=True)

df5
print('# movies for which High_relevance_tag is not available:{}'.format(df5['High_relevance_tag'].isnull().sum()))
('max number of genres for a single movie {}'.format(max(df5['genres'].str.split('|').apply(len))))
# #Loop to find each genre used in the dataset.

# genres_lis=set({})

# for i in range(len(df2)):

#     for j in range(len(df2['genres'].str.split('|')[i])):

# #         print(i,j)

#         genres_lis.add(df2['genres'].str.split('|')[i][j])

        

# genres_lis=list(genres_lis)

# print('Genres:{}'.format(genres_lis))

genres_lis=['Romance', 'Fantasy', 'Animation', 'Action', 'Film-Noir', 'Adventure', 'Horror', 'IMAX', 'Musical', 'War', 'Drama', 'Comedy', 'Crime', 'Mystery', '(no genres listed)', 'Documentary', 'Western', 'Thriller', 'Children', 'Sci-Fi']
print('Number of different genres present in the dataset are {}'.format(len(genres_lis)))
# Loop to creat columns for each genre



for i in range(len(genres_lis)):

    df5[genres_lis[i]]=df5['genres'].str.contains(genres_lis[i])

    
df5.head()
for genre in genres_lis:

    df5.groupby(genre)['rating'].mean().plot.bar()

    plt.ylabel('Avg. Rating')

    plt.show()

#  Figures show Avg rating for different genres.
# Loop to find avg rating on the basis of each genre.

avg_rating=[]

for genre in genres_lis:

    avg_rating.append(df2[df2['genres'].str.contains(genre)]['rating'].mean())

# avg_rating

plt.figure(figsize=(8,5))

sns.barplot(genres_lis, avg_rating,palette='Set1')

plt.xticks(rotation='vertical')

plt.ylabel('Avg. Rating')

plt.xlabel('Genres')

plt.title('Avg. Rating W.R.T Genres', fontsize=16)
for genre in genres_lis:

#     print(genre)

    df2[df2['genres'].str.contains(genre)][['Year','movieId']].groupby('Year').count().reset_index().sort_values(by='Year').plot('Year', 'movieId')

    plt.legend('')

    plt.ylabel('# of movies/year in {} genre'.format(genre))

#     plt.xticks()

    plt.show()
dfx=df5.dropna()

dfx['Year']=dfx['Year'].astype('int64')

plt.figure(figsize = (10,8))

sns.heatmap(dfx[['rating', 'Year', 'High_relevance_tagId','Romance', 'Romance', 'Fantasy', 'Animation',

       'Action', 'Film-Noir', 'Adventure', 'Horror', 'IMAX', 'Musical', 'War',

       'Drama', 'Comedy', 'Crime', 'Mystery', '(no genres listed)',

       'Documentary', 'Western', 'Thriller', 'Children', 'Sci-Fi']].corr(), annot=False)
dfx[['rating', 'Year', 'High_relevance_tagId','Romance', 'Romance', 'Fantasy', 'Animation',

       'Action', 'Film-Noir', 'Adventure', 'Horror', 'IMAX', 'Musical', 'War',

       'Drama', 'Comedy', 'Crime', 'Mystery', '(no genres listed)',

       'Documentary', 'Western', 'Thriller', 'Children', 'Sci-Fi']].corr()['rating'].sort_values(ascending=False)
# Let's suppose on a weekend you are with your girlfriend, and your girl likes to watch new romantic movies, but you like comedy and obviously both 

#  would like to see some high rated movies.

is_romantic=df5['genres'].str.contains('Romance')

is_comedy=df5['genres'].str.contains('Comedy')

is_highly_rated=df5['rating']>4

is_new=df5['Year'].dropna().astype('int64')>1990

df5[is_romantic & is_comedy & is_highly_rated & is_new][['movieId','title','genres','rating','Year','High_relevance_tag']]