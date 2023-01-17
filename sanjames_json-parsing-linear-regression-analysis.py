#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import json
credits = pd.read_csv('../input/tmdb_5000_credits.csv')
movies = pd.read_csv('../input/tmdb_5000_movies.csv')
credits.shape
movies.shape
credits.head(2)
credits.info()
movies.head(2)
movies.info()
allmovies_df = pd.merge(left=movies,right=credits, left_on='id', right_on='movie_id', suffixes=('_left', '_right'))
allmovies_df.shape
allmovies_df.info()
# Both id and movie_id refers to movie_id. Also title_right and title_left refers to movie title
#Drop the column 'id' from the dataframe allmovies_df. 
allmovies_df.drop(['id', 'title_right'], axis=1, inplace=True)
allmovies_df = allmovies_df.rename(columns={'title_left': 'title'})
allmovies_df.head(2)
#Change the order of the dataframe allmovies_df
allmovies_df = allmovies_df[['movie_id', 'budget', 'title', 'original_title', 'status', 'tagline', 'release_date', 'runtime', 
               'genres', 'production_companies', 'production_countries', 'popularity', 'revenue', 'vote_average',
               'vote_count', 'cast', 'crew', 'homepage', 'keywords', 'original_language', 'overview', 'spoken_languages'
             ]]
allmovies_df.head(2)
allm = allmovies_df.copy() #just for backup
#parse json input
#NOTE: I'm parsing crew, cast and production companies separately.
json_columns = ['genres', 'keywords', 'production_countries', 'spoken_languages']
for column in json_columns:
    allmovies_df[column] = allmovies_df[column].apply(json.loads, encoding="utf-8")
allmovies_df['crew'] = allmovies_df['crew'].apply(json.loads, encoding="utf-8")    
allmovies_df['cast'] = allmovies_df['cast'].apply(json.loads, encoding="utf-8")    
allmovies_df['production_companies'] = allmovies_df['production_companies'].apply(json.loads, encoding="utf-8")
def process_jsoncols(colname):
    jsoncollist=[]
    for x in colname:
        jsoncollist.append(x['name'])
    return jsoncollist
for colname in json_columns:
    allmovies_df[colname] = allmovies_df[colname].apply(process_jsoncols)
allmovies_df[['genres', 'keywords', 'production_countries', 'spoken_languages']].head()
allmovies_df['production_companies'] = allmovies_df['production_companies'].apply(process_jsoncols)
allmovies_df['production_companies'].head(2)
for index,x in zip(allmovies_df.index,allmovies_df['cast']):
    castlist=[]
    for i in range(len(x)):
        if (x[i]['order'] < 1):
            castlist.append((x[i]['name']))
    allmovies_df.loc[index,'cast']=str(castlist)
allmovies_df['cast'].head(2)
#allmovies_df['cast'] = allmovies_df['cast'].str.strip('[]').str.replace("'",'').str.replace('"','').str.replace(' ','').str.replace(',Jr.','Jr.')
allmovies_df['cast'] = allmovies_df['cast'].str.strip('[]').str.replace("'",'').str.replace('"','').str.replace(' ','')
#Checking to see all the information is correct.
allmovies_df[allmovies_df['cast'].isnull()]
allmovies_df['cast'].head(2)
for index,x in zip(allmovies_df.index,allmovies_df['crew']):
    crewlist=[]
    for i in range(len(x)):
        if (x[i]['job'] == 'Director'):
            crewlist.append((x[i]['name']))
    allmovies_df.loc[index,'crew']=str(crewlist)
#def process_jsoncol_crew(colname):
#    crewlist=[]
#    for x in colname:
#        if x['job'] == 'Director':
#            crewlist.append(x['name'])
#            return crewlist
#allmovies_df['crew'] = allmovies_df['crew'].apply(process_jsoncol_crew)
#for index,x in zip(allmovies_df.index,allmovies_df['crew']):
#    crewlist=[]
#    for i in range(len(x)):
#        if (x[i]['job'] == 'Director'):
#            print(x[i]['job'])
#            crewlist.append((x[i]['job']))
#            print(crewlist)
#    allmovies_df.loc[index,'crew']=str(crewlist)
allmovies_df['crew'].head(2)
allmovies_df['crew'].isnull().sum()
#allmovies_df['cast'] = allmovies_df['cast'].str.strip('[]').str.replace("'",'').str.replace('"','').str.replace(' ','').str.replace(',Jr.','Jr.')
allmovies_df['crew'] = allmovies_df['crew'].str.strip('[]').str.replace("'",'').str.replace('"','').str.replace(' ','')
allmovies_df['crew'].head(2)
listcols = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
for colname in listcols:
    allmovies_df[colname] = allmovies_df[colname].apply(lambda x: ','.join(map(str, x)))
allmovies_df.head(2)
from datetime import datetime
allmovies_df['release_date'] = pd.to_datetime(allmovies_df['release_date'])
allmovies_df['release_year'] = allmovies_df['release_date'].dt.year
allmovies_df['release_month'] = allmovies_df['release_date'].dt.month
allmovies_df[['release_year','release_month']].head(2)
#another backup
afterjson = allmovies_df.copy()
movies_genres = pd.DataFrame(allmovies_df[['movie_id', 'budget', 'title','release_year', 'release_month','genres','revenue','vote_average','vote_count','original_language']])
movies_genres.head(2)
genres_list = set()
for sstr in allmovies_df['genres'].str.split(','):
    genres_list = set().union(sstr, genres_list)
genres_list = list(genres_list)
genres_list.remove('')
genres_list
#pd.Series(' '.join(movies_genres['genres']).split('|')).value_counts()
#pd.Series(' '.join(movies_genres['genres']).lower().split()).value_counts()[:10]
#Transforming categorical to one hot encoding
for genres in genres_list:
    movies_genres[genres] = movies_genres['genres'].str.contains(genres).apply(lambda x:1 if x else 0)
movies_genres.head(2)
genre_count = []
for genre in genres_list:
    genre_count.append([genre, movies_genres[genre].values.sum()])
names = ['genrename','genrecount']
genre_df = pd.DataFrame(data=genre_count, columns=names)
genre_df.sort_values("genrecount", inplace=True, ascending=False)
genre_df.head()
labels=genre_df.genrename
plt.subplots(figsize=(10, 10))
genre_df.genrecount.plot.pie(labels = labels, autopct='%1.1f%%', shadow=False)
plt.subplots(figsize=(10, 10))
genre_df['genrecount'].plot.bar( align='center', alpha=0.5, color='red')
y_pos = np.arange(len(labels))
#plt.yticks(y_pos, labels)
plt.xticks(y_pos, labels)
plt.ylabel('Genres Count')
movies_cast = allmovies_df[['movie_id', 'budget', 'title','release_year', 'release_month','cast','revenue','vote_average','vote_count','original_language']]
movies_cast[movies_cast['cast'].isnull()]
cast_list = list(movies_cast['cast'])
cast_list
def count_elements(lst):
    elements = {}
    for elem in lst:
        if elem in elements.keys():
            elements[elem] +=1
        else:
            elements[elem] = 1
    return elements
castcount = count_elements(cast_list)
top30_cast = sorted(castcount, key=castcount.get, reverse=True)[1:30]
top30_cast
for cast in top30_cast:
    movies_cast[cast] = movies_cast['cast'].str.contains(cast).apply(lambda x:1 if x else 0)
movies_cast.head(2)
cast_count = []
for cast in top30_cast:
    cast_count.append([cast, movies_cast[cast].values.sum()])
names = ['castname','castcount']
cast_df = pd.DataFrame(data=cast_count, columns=names)
cast_df.sort_values("castcount", inplace=True, ascending=False)
cast_df.head()
cast_labels = cast_df.castname[cast_df['castcount']>15]
plt.subplots(figsize=(10, 10))
cast_df.castcount[cast_df['castcount']>15].plot.bar( align='center', alpha=0.5)
y_pos = np.arange(len(cast_labels))
#plt.yticks(y_pos, cast_labels)
plt.xticks(y_pos, cast_labels)
plt.ylabel('cast Count')
movies_crew = allmovies_df[['movie_id','budget','title','release_year','release_month','crew','revenue','vote_average','vote_count','original_language']]
#movies_crew = movies_crew[movies_crew['crew'].notnull()]
movies_crew.index = pd.RangeIndex(len(movies_crew.index))
movies_crew.isnull().sum()
crew_list = list(movies_crew['crew'])
crew_list
crewcount = count_elements(crew_list)
top30_crew = sorted(crewcount, key=crewcount.get, reverse=True)[1:30]
for crew in top30_crew:
    movies_crew[crew] = movies_crew['crew'].str.contains(crew).apply(lambda x:1 if x else 0)
movies_crew.head(3)
crew_count = []
for crew in top30_crew:
    crew_count.append([crew, movies_crew[crew].values.sum()])
names = ['crewname','crewcount']
crew_df = pd.DataFrame(data=crew_count, columns=names)
crew_df.sort_values("crewcount", inplace=True, ascending=False)
crew_df.head()
crew_labels = crew_df.crewname[crew_df['crewcount']>9]
plt.subplots(figsize=(10, 10))
crew_df.crewcount[crew_df['crewcount']>9].plot.bar( align='center', alpha=0.5, color='purple')
y_pos = np.arange(len(crew_labels))
#plt.yticks(y_pos, crew_labels)
plt.xticks(y_pos, crew_labels)
plt.ylabel('crew Count')
movies_production_companies = allmovies_df[['movie_id','budget','title','release_year','release_month','production_companies','revenue','vote_average','vote_count','original_language']]
movies_production_companies.head(2)
top30_production_companies = ['Paramount Pictures','Columbia Pictures','Twentieth Century Fox Film Corporation','Metro-Goldwyn-Mayer (MGM)',
               'Marvel Studios','Walt Disney Pictures','Walt Disney','Walt Disney Animation Studios',
               'Walt Disney Studios Motion Pictures','Warner Bros.','Universal Pictures','Universal Studios',
               'Jerry Bruckheimer Films','Pixar Animation Studios','Relativity Media','Lucasfilm',
               'RKO Radio Pictures','New Line Cinema','Miramax Films','DreamWorks','DreamWorks SKG']
for production_companies in top30_production_companies:
    movies_production_companies[production_companies] = movies_production_companies['production_companies'].str.contains(production_companies).apply(lambda x:1 if x else 0)
movies_production_companies.head(2)
production_companies_count = []
for production_companies in top30_production_companies:
    production_companies_count.append([production_companies, movies_production_companies[production_companies].values.sum()])
production_companies_count
names = ['production_companiesname','production_companiescount']
production_companies_df = pd.DataFrame(data=production_companies_count, columns=names)
production_companies_df.sort_values("production_companiescount", inplace=True, ascending=False)
production_companies_df
production_companies_labels = production_companies_df.production_companiesname[production_companies_df['production_companiescount']>1]
production_companies_df.head()
plt.subplots(figsize=(10, 10))
production_companies_df.production_companiescount[production_companies_df['production_companiescount']>1].plot.bar( align='center', alpha=0.5, color='red')
y_pos = np.arange(len(production_companies_labels))
#plt.yticks(y_pos, production_companies_labels)
plt.xticks(y_pos, production_companies_labels)
plt.ylabel('production_companies Count')
plt.figure(figsize=(12,8))
sns.countplot(x='release_year', data=movies_genres, color='red')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Movies released per year', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Movies released by year", fontsize=15)
plt.show()
movies_genres[['release_year', 'release_month']].groupby(['release_year'], as_index=False).count().sort_values(by='release_year', ascending=False)
movies_genres[['release_month', 'release_year']].groupby(['release_month'], as_index=False).count().sort_values(by='release_year', ascending=False)
plt.figure(figsize=(12,8))
sns.countplot(x='release_month', data=movies_genres, color='red')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Movies released per month', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Movies released by year", fontsize=15)
plt.show()
movies_genres['revenue'].plot.hist(alpha=0.5, bins=20)
plt.title('Histogram of the Revenue')
plt.xlabel("Revenue")
plt.ylabel("Frequency") 
movies_genres['budget'].plot.hist(alpha=0.5, bins=20)
plt.title('Histogram of the Revenue')
plt.xlabel("Revenue")
plt.ylabel("Frequency") 
# Importing modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
genre_corr = movies_genres.corr()
genre_corr['revenue'].sort_values()
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
movies_genres['budget'] = MinMaxScaler().fit_transform(movies_genres['budget'])
movies_genres['vote_average'] = MinMaxScaler().fit_transform(movies_genres['vote_average'])
movies_genres['vote_count'] = MinMaxScaler().fit_transform(movies_genres['vote_count'])

x = movies_genres[['vote_count','budget','Adventure', 'Fantasy', 'Action', 'Animation', 'vote_average', 'Family', 
                   'Science Fiction', 'Drama']]
x.head(3)
y = movies_genres['revenue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
#Predict Output
lin_predicted = linear.predict(X_test)

linear_score = round(linear.score(X_train, y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
pd.DataFrame(list(zip(x.columns, linear.coef_)), columns = ['features', 'coefficients'])
#Regression plot between budget and revenue
plt.figure(figsize=(8,8))
sns.regplot(x=movies_genres["budget"], y=movies_genres["revenue"], fit_reg=True)
movies_genres[movies_genres['revenue'] > 2500000000]
plt.figure(figsize=(8,8))
sns.boxplot(x=movies_genres["Animation"], y=movies_genres["revenue"])
plt.figure(figsize=(8,8))
sns.regplot(x=movies_genres["vote_count"], y=movies_genres["revenue"], fit_reg=True)
plt.figure(figsize=(8,8))
sns.boxplot(x=movies_genres["Drama"], y=movies_genres["revenue"])
mov_g = movies_genres[movies_genres['revenue'] < 2500000000]
mov_g.shape
x = mov_g[['budget']]
y = mov_g['revenue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
#Predict Output
lin_predicted = linear.predict(X_test)

linear_score = round(linear.score(X_train, y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
x = mov_g[['budget', 'vote_count']]
y = mov_g['revenue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
#Predict Output
lin_predicted = linear.predict(X_test)

linear_score = round(linear.score(X_train, y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
mov_g.info()
len(mov_g)
mov_g[mov_g.release_year.isnull()]
mov_g = mov_g.dropna(axis=0, how='any')
mov_g.head()
x = mov_g[['budget','release_year','release_month','vote_count','Animation','Thriller','Family',
           'Adventure','Western','War','Drama','Action','Mystery','Science Fiction','Documentary','Foreign','TV Movie','Fantasy',
           'Music','History','Horror','Romance','Crime','Comedy']]
y = mov_g['revenue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
#Predict Output
lin_predicted = linear.predict(X_test)

linear_score = round(linear.score(X_train, y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
pd.DataFrame(list(zip(x.columns, linear.coef_)), columns = ['features', 'coefficients'])
sns.pairplot(mov_g, x_vars=['budget','release_year','vote_count'], y_vars='revenue', size=7, aspect=0.7, kind='reg')
x = mov_g[['budget','release_year','release_month','vote_average','vote_count','Animation','Thriller','Family',
           'Adventure','Western','War','Drama','Action','Mystery','Science Fiction','Documentary','Foreign','TV Movie','Fantasy',
           'Music','History','Horror','Romance','Crime','Comedy']]
y = mov_g['revenue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
#Predict Output
lin_predicted = linear.predict(X_test)

linear_score = round(linear.score(X_train, y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
