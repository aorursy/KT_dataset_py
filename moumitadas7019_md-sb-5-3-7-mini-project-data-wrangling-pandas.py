#!pip install --upgrade pip
!pip install py4j==0.10.7
#!pip install pandas==0.23
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
pd.__version__
#movies = pd.read_csv('../input/md-imdb/titles.csv', compression='bz2')

movies = pd.read_csv('../input/md-imdb/titles.csv')
movies.info()
movies.head()
cast = pd.read_csv('../input/md-imdb/cast.csv')
cast.info()
cast.head(10)
release_dates = pd.read_csv('../input/md-imdb/release_dates.csv', parse_dates=['date'], infer_datetime_format=True)
release_dates.info()
release_dates.head()
len(movies)
batman_df = movies[movies.title == 'Batman']
print('Total Batman Movies:', len(batman_df))
batman_df
batman_df = movies[movies.title.str.contains('Batman', case=False)]
print('Total Batman Movies:', len(batman_df))
batman_df.head(10)
batman_df.sort_values(by=['year'], ascending=True).iloc[:15]
harry_df = movies[movies.title.str.contains('Harry Potter', case=False)]
print('Total Harry Potter Movies:', len(harry_df))
harry_df.head(10)
harry_df.sort_values(by=['year'], ascending=False)
len(movies[movies.year == 2017])
len(movies[movies.year == 2015])
#(movies[(movies.year >= 2000)&(movies.year <= 2018)])
len(movies[(movies.year >= 2000) & (movies.year <= 2018)])
hamlet_df = movies[movies.title=='Hamlet']
len(hamlet_df)
hamlet_df[hamlet_df.year >= 2000].sort_values(by=['year'], ascending=False)
sup_cast = cast[(cast.title == 'Inception') & (pd.isnull(cast.n))]
len(sup_cast)
main_cast = cast[(cast.title == 'Inception') & ~(pd.isnull(cast.n))]
len(main_cast)
top_cast = (cast[(cast.title == 'Inception') & ~(pd.isnull(cast.n))].sort_values(by='n', ascending=True)).iloc[:10]
top_cast
cast[cast.character == 'Albus Dumbledore']
cast[cast.character == 'Albus Dumbledore'][['name']].drop_duplicates()
len(cast[cast.name=='Keanu Reeves'])

cast[(cast.name=='Keanu Reeves') & ( cast.n==1) & (cast.year >=1999)].sort_values(by='year', ascending=False)

cast.head()
(cast[cast.year.between(1950, 1960)][['type', 'name']]
.groupby('type')
.count()
.reset_index()
.rename({'name': 'freq'}, axis=1))
(cast[cast.year.between(2007, 2017)][['type', 'name']]
.groupby('type')
.count()
.reset_index()
.rename({'name': 'freq'}, axis=1))
cast[(cast.year >= 2000) & (cast.n == 1)]
cast[(cast.year >= 2000) & (cast.n != 1) & ~(pd.isnull(cast.n))]
cast[(cast.year >= 2000) & (pd.isnull(cast.n))]
top_ten = movies.title.value_counts()[:10]
top_ten
top_ten.plot(kind='barh')
movies.head()
movies[movies.year // 10 == 200]['year'].value_counts()[:3]
(movies.year // 10 * 10).value_counts().sort_index().plot(kind='bar')
cast.character.value_counts()[:10]
cast[cast.character == 'Herself']['name'].value_counts()[:10]
cast[cast.character == 'Himself']['name'].value_counts()[:10]
cast[cast.character.str.startswith('Zombie')].character.value_counts().head(10)
cast[cast.character.str.startswith('Police')].character.value_counts().head(10)
cast[cast.name=='Keanu Reeves'].year.value_counts().sort_index().plot(kind='barh')
keanu = cast[(cast.name == 'Keanu Reeves') & (pd.notnull(cast.n))][['year', 'n']].sort_values('year')
keanu.plot(x='year', y='n', kind='scatter')
hamlet = (movies[movies.title == 'Hamlet']
          .groupby(movies.year // 10 * 10)
          .count()
          .rename({'title': 'count'}, axis=1))['count']
hamlet.plot(kind='bar')
(cast[(cast.year.between(1960, 1969)) & (cast.n == 1)]
.groupby(['year', 'type'])
.count()[['title']]
.rename({'title': 'count'}, axis=1))
(cast[(cast.year.between(2000, 2009)) & (cast.n == 1)]
.groupby(['year', 'type'])
.count()[['title']]
.rename({'title': 'count'}, axis=1))
frank = (cast[cast.name == 'Frank Oz']
         .groupby(['year', 'title'])
         .count()[['name']]
         .rename({'name': 'freq'}, axis=1)
         .sort_values(by=['year'], ascending=True))
frank[frank.freq > 1]
frank = (cast[cast.name == 'Frank Oz']
         .groupby(['character'])
         .count()[['name']]
         .rename({'name': 'freq'}, axis=1)
         .sort_values(by=['freq'], ascending=False))
frank[frank.freq > 1]
christmas = release_dates[(release_dates.title.str.contains('Christmas')) & (release_dates.country == 'USA')]
christmas.date.dt.month.value_counts().sort_index().plot(kind='bar')
summer = release_dates[(release_dates.title.str.contains('Summer')) & (release_dates.country == 'USA')]
summer.date.dt.month.value_counts().sort_index().plot(kind='bar')
action = release_dates[(release_dates.title.str.contains('Action')) & (release_dates.country == 'USA')]
action.date.dt.dayofweek.value_counts().sort_index().plot(kind='bar')
us = release_dates[release_dates.country == 'USA']
keanu = cast[(cast.name == 'Keanu Reeves') & (cast.n == 1)]
(keanu.merge(us, how='inner', on=['title', 'year'])
      .sort_values('date')) 
us = release_dates[release_dates.country == 'USA']
keanu = cast[(cast.name == 'Keanu Reeves')]
keanu = (keanu.merge(us, how='inner', on=['title', 'year'])
              .sort_values('date'))
keanu.date.dt.month.value_counts().sort_index().plot(kind='bar')
us = release_dates[release_dates.country == 'USA']
ian = cast[(cast.name == 'Ian McKellen')]
ian = (ian.merge(us, how='inner', on=['title', 'year'])
              .sort_values('date'))
ian.date.dt.year.value_counts().sort_index().plot(kind='bar')