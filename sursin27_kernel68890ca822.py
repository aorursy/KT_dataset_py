import pandas as pd
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

fruit_sales = pd.DataFrame({'Apples':pd.Series(data=[35,21],index=['2017 Sales']),'Bananas':pd.Series(data=[41,34],index=['2018 Sales'])})



q2.check()

fruit_sales
ser = pd.Series([100,2,3],['a','b','c'])

ser
ser = pd.Series(data=[100, 200, 300, 400, 500], index=['tom', 'bob', 'nancy', 'dan', 'eric'])
ser
ser.index
ser[['nancy','dan']]
ser['nancy']
ser[[4, 3, 1]]
'bob' in ser
ser
ser['dan']='suresh'
ser * 2
ser['dan']=2000
ser
ser ** 2

#ser[['tom','bob']]**2

d = {'one' : pd.Series([100., 200., 300.], index=['apple', 'ball', 'clock']),

     'two' : pd.Series([111., 222., 333., 4444.], index=['apple', 'ball', 'cerill', 'dancy'])}
df = pd.DataFrame(d)

#print(df)

df
df.index
df.columns
pd.DataFrame(d, index=['dancy', 'ball', 'apple'])
pd.DataFrame(d, index=['dancy', 'ball', 'apple'], columns=['two','five'])
data = [{'alex': 1, 'joe': 2}, {'ema': 5, 'dora': 10, 'alice':20}]
pd.DataFrame(data)
pd.DataFrame(data, index=['orange', 'red'])
pd.DataFrame(data, columns=['joe', 'dora','alice','a'])
df
df['one']
df['three'] = df['one'] * df['two']

df
df['flag'] = df['one'] > 250

df
three = df.pop('three')
three
df
del df['one']
df
df.insert(1, 'copy_of_onee', df['two'])

df
df['one_upper_half'] = df['two'][:2]

df
# Note: Adjust the name of the folder to match your local directory



!ls ./movielens
!cat ./movielens/movies.csv
movies = pd.read_csv('../input/movielens-20m-dataset/movie.csv', sep=',')

#print(type(movies))

movies.head()
# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970



tags = pd.read_csv('../input/movielens-20m-dataset/genome_tags.csv', sep=',')

tags.head()
#ratings = pd.read_csv('./movielens/ratings.csv', sep=',', parse_dates=['timestamp'])

ratings = pd.read_csv('../input/movielens-20m-dataset/rating.csv', sep=',')

ratings.head()
ratings
# For current analysis, we will remove timestamp (we will come back to it!)



del ratings['timestamp']

#del tags['timestamp']
row = ratings.iloc[5]

row

col = ratings.ix[5]

col
#Extract 0th row: notice that it is infact a Series



row_0 = tags.iloc[5]

row_0

#type(row_0)
print(row_0)
row_0.index
row_0['tagId']
'rating' in row_0
row_0.name
row_0 = row_0.rename('first_row')

row_0.name
tags.head(10)
tags.index
tags.columns
# Extract row 0, 11, 2000 from DataFrame



tags.iloc[ [0,11,1127] ]
ratings
ratings['rating'].describe()
ratings.describe()
ratings['rating'].mean()
ratings.mean()
ratings.min()
ratings['rating'].max()
ratings['rating'].std()
ratings['rating'].mode()
ratings.corr()
filter_1 = ratings['rating'] > 5



filter_1.any()
filter_1
filter_2 = ratings['rating'] > 0

filter_2.all()
movies.shape
#is any row NULL ?



movies.isnull().any()
ratings = pd.read_csv('../input/movielens-20m-dataset/rating.csv', sep=',')
ratings.shape
#is any row NULL ?



ratings.isnull().any()
tags.shape
#is any row NULL ?



tags.isnull().any()
tags = tags.dropna()
#Check again: is any row NULL ?



tags.isnull().any()
tags.shape
%matplotlib inline



ratings.hist(column='rating', figsize=(15,10))
ratings.boxplot(column='rating',figsize=(15,10))
tags['tag'].head()
movies[['title','genres']].head()

ratings[1000:1010]
tags
tag_counts = tags['tag'].value_counts()

tag_counts.head(50)
tag_counts[:10].plot(kind='bar', figsize=(15,10))
is_highly_rated = ratings['rating'] >= 4.0



ratings[is_highly_rated][-5:]
is_animation = movies['genres'].str.contains('Animation')



movies[is_animation][5:15]
war = movies['genres'].str.contains('War')

movies[war]
movies[is_animation].head(15)
ratings_count = ratings[['movieId','rating']].groupby('rating').count()

ratings_count
average_rating = ratings[['movieId','rating']].groupby('movieId').mean()

average_rating.tail()
movie_count = ratings[['movieId','rating']].groupby('movieId').count()

movie_count.head()
movie_count = ratings[['movieId','rating']].groupby('movieId').count()

movie_count.tail()
tags.head()
tags['movieId']=tags['tagId']

tags
movies.head()
t = movies.merge(tags, on='movieId', how='inner')

t.head()
avg_ratings = ratings.groupby('movieId', as_index=False).mean()

del avg_ratings['userId']

avg_ratings.head()
box_office = movies.merge(avg_ratings, on='movieId', how='inner')

box_office.tail()
is_highly_rated = box_office['rating'] >= 4.0



box_office[is_highly_rated][-5:]
is_comedy = box_office['genres'].str.contains('Comedy')



box_office[is_comedy][:5]
box_office[is_comedy & is_highly_rated][-5:]
movies.head()
movie_genres = movies['genres'].str.split('|', expand=True)
movie_genres[:10]
movie_genres['isComedy'] = movies['genres'].str.contains('Comedy')
movie_genres[:10]
movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)
movies.tail()
tags = pd.read_csv('../input/movielens-20m-dataset/tag.csv', sep=',')
tags.dtypes
tags.head(5)
tags['parsed_time'] = pd.to_datetime(tags['timestamp'], unit='s')


tags['parsed_time'].dtype
tags.head(2)
greater_than_t = tags['parsed_time'] > '2015-02-01'



selected_rows = tags[greater_than_t]



tags.shape, selected_rows.shape
tags.sort_values(by='parsed_time', ascending=True)[:10]
average_rating = ratings[['movieId','rating']].groupby('movieId', as_index=False).mean()

average_rating.tail()
joined = movies.merge(average_rating, on='movieId', how='inner')

joined.head()

joined.corr()
yearly_average = joined[['year','rating']].groupby('year', as_index=False).mean()

yearly_average[:10]
yearly_average[-20:].plot(x='year', y='rating', figsize=(15,10), grid=True)