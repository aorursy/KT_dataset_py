# Import libraries

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline



import json

import ast



from wordcloud import WordCloud, STOPWORDS

import nltk

from nltk.probability import FreqDist



from collections import Counter



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected=True)
# Load train and test data

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')



# Print shape of train and test data

print('Train data shape: ', train.shape)

print('Test data shape: ', test.shape)
# Lets inspect some sample rows from train data

train.sample(3)



# Lots of features are dictionaries so we will need to take closer look to them.

# Some features like homepage cound be binarized.
# Describe train data

train.describe()
# Inspect test data

test.sample(3)
# Check missing data

pd.concat([train.isnull().sum(), test.isnull().sum()], keys=['train','test'], axis=1, sort=False)
# How many missing values have feature in train and test set.

def how_many_missing_values(feature):

    print('Missing {} in train set: {}/{} ({}%)'.format(feature,

                                                        train[feature].isnull().sum(), 

                                                        train[feature].size,

                                                        train[feature].isnull().sum()/train[feature].size))



    print('Missing {} in test set: {}/{} ({}%)'.format(feature,

                                                       test[feature].isnull().sum(), 

                                                       test[feature].size,

                                                       test[feature].isnull().sum()/test[feature].size))
# Remove id feature

#train.drop(labels=['id'], axis=1, inplace=True)

#test.drop(labels=['id'], axis=1, inplace=True)
# Count uniqe values for belongs_to_collection

train['belongs_to_collection'].value_counts().head(10)
# Get the collection name

def get_collection_name(dataset):

    return dataset['belongs_to_collection'].fillna(0).apply(lambda x: eval(x)[0]['name'] if x != 0 else 0)



# Check if movie belongs to collection

def is_belonging_to_collection(dataset):

    return dataset['belongs_to_collection'].fillna(0).apply(lambda x: 1 if x != 0 else 0)



# New feature collection name

train['collection_name'] = get_collection_name(train)

test['collection_name'] = get_collection_name(test)



# Change belongs_to_collection to bool

train['belongs_to_collection'] = is_belonging_to_collection(train)

test['belongs_to_collection'] = is_belonging_to_collection(test)
sns.countplot(train['belongs_to_collection'])

train['belongs_to_collection'].value_counts()
# train['collection_name'].value_counts()

train[train['belongs_to_collection'] == 1]['collection_name'].value_counts().head(10).plot(kind='bar')
sns.boxplot(x='belongs_to_collection', y='revenue', data=train)
(train[train['belongs_to_collection'] == 1].groupby('collection_name')['revenue']

 .agg('sum').sort_values(ascending=False).head(20).plot(kind='bar'))
g = sns.catplot(x='collection_name', y='revenue', data=train[train['belongs_to_collection'] == 1],

                order=train[train['belongs_to_collection'] == 1].groupby('collection_name')['revenue']

                .agg('sum').sort_values(ascending=False).head(20).index, aspect=3)

g.set_xticklabels(rotation=90)
(train[train['belongs_to_collection'] == 1].groupby('collection_name')['revenue']

 .agg('mean').sort_values(ascending=False).head(20).plot(kind='bar'))
train[['title','budget']].sort_values(by='budget',ascending=False).head(10)
sns.lmplot(x='budget', y='revenue', data=train, height=5, aspect=2)
sns.boxplot(x='belongs_to_collection', y='budget', data=train)
# Create new features for ratio and profit

train = train.assign(ratio = lambda df: df['revenue']/df['budget'])

train = train.assign(profit = lambda df: df['revenue']-df['budget'])
# Get Top 10 movies with biggest revenue/budget ratio.

train[train['budget'] > 0][['title','budget','revenue','ratio','profit']].sort_values(by='ratio', ascending=False).head(10)
# Get Top 10 movies with the biggest budget

train[['title','budget','revenue','ratio','profit']].sort_values(by='budget', ascending=False).head(10)
# Plot movies with the biggest profit

train.sort_values(by='profit', ascending=False).head(10).plot(x='title', y='profit', kind='barh')
# Get movies that made profit

train['made_profit'] = train['profit'] > 0

sns.countplot(x='made_profit', data=train)

train['made_profit'].value_counts()
# Show top five rows

train['genres'].head(5)
# Parse json data

def parse_json(x):

    try: return json.loads(x.replace("'", '"'))

    except: return ''

    

# Parse genres

train['genres'] = train['genres'].apply(parse_json)
# Get number of genres for movies

train['genres'].apply(len).hist(bins=8)

print('Mean: ', train['genres'].apply(len).mean())
# Get the number of genres as a new feature

train['number_of_genres'] = train['genres'].apply(len)
# Get budget, revenue, profit data for movies based on number of genres

(train.groupby('number_of_genres')[['budget','revenue','profit']].agg(['mean','median','count'])

 .sort_values(by=('profit','mean'), ascending=False))
# Get top 4 genres

def top_genres(genres):

    if len(genres) == 1:

        return pd.Series([genres[0]['name'], '', '', ''], 

                         index=['genre1', 'genre2', 'genre3', 'genre4'])

    if len(genres) == 2:

        return pd.Series([genres[0]['name'], genres[1]['name'], '', ''], 

                         index=['genre1', 'genre2', 'genre3', 'genre4'])

    if len(genres) == 3:

        return pd.Series([genres[0]['name'], genres[1]['name'], genres[2]['name'], ''], 

                         index=['genre1', 'genre2', 'genre3', 'genre4'])

    if len(genres) > 3:

        return pd.Series([genres[0]['name'], genres[1]['name'], genres[2]['name'], genres[3]['name']], 

                         index=['genre1', 'genre2', 'genre3', 'genre4'])

    return pd.Series(['','','',''], index=['genre1','genre2','genre3','genre4'])



train[['genre1', 'genre2', 'genre3', 'genre4']] = train['genres'].apply(top_genres)
genres_df = pd.concat([train['genre1'].value_counts(), train['genre2'].value_counts(), 

                       train['genre3'].value_counts(), train['genre4'].value_counts()], 

                      axis=1, sort=False)

genres_df['sum'] = genres_df.sum(axis=1)



# Show genres data

genres_df
genres_df[~genres_df.index.isin([''])]['sum'].plot(kind='pie', figsize=(9,9))
genres_df[~genres_df.index.isin([''])][['genre1','genre2','genre3','genre4']].plot(kind='pie', subplots='True', 

                                                                                   figsize=(15,15), layout=(2,2), 

                                                                                   legend=False, 

                                                                                   title=['Genre 1','Genre 2',

                                                                                          'Genre 3','Genre 4'])
sns.catplot(x='genre1', y='revenue', data=train, 

            order=train.groupby('genre1')['revenue'].mean().sort_values(ascending=False).index, 

            kind='box', height=4, aspect=5)
sns.catplot(x='genre1', y='budget', data=train, 

            order=train.groupby('genre1')['budget'].mean().sort_values(ascending=False).index, 

            kind='box', height=4, aspect=5)
sns.catplot(x='genre1', y='profit', data=train, 

            order=train.groupby('genre1')['profit'].mean().sort_values(ascending=False).index, 

            kind='box', height=4, aspect=5)
train['homepage'].notnull().value_counts().plot(kind='pie', autopct='%1.1f%%')
# Create new feature describing if a movie have a homepage.

train['has_homepage'] = train['homepage'].notnull()

test['has_homepage'] = test['homepage'].notnull()
# Remove homepage feature

#train.drop(labels=['homepage'], axis=1, inplace=True)

#test.drop(labels=['homepage'], axis=1, inplace=True)
sns.boxplot(x='has_homepage', y='revenue', data=train)
sns.boxplot(x='has_homepage', y='budget', data=train)
plt.figure(figsize=(15,5))

sns.lineplot(x='budget', y='profit', hue='has_homepage', data=train, style='has_homepage')
# Delete imdb_id feature

# train.drop(labels=['imdb_id'], axis=1, inplace=True)

# test.drop(labels=['imdb_id'], axis=1, inplace=True)
plt.figure(figsize=(15,5))

sns.countplot(x='original_language', data=train, order=train['original_language'].value_counts().index)
(train.groupby('original_language')['revenue'].mean().sort_values(ascending=False)

 .plot(kind='barh', figsize=(12,6), title='Which original language had the highest revenue?'))
(train.groupby('original_language')['profit'].mean().sort_values(ascending=False)

 .plot(kind='barh', figsize=(12,6), title='How original language affects profit?'))
# Get all orginal titles

text = ' '.join(train['original_title'])



# Generate WordCloud from titles

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)



# Display wordcloud

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# Get top 10 words in orginal title

stopwords = set(STOPWORDS)

words = text.lower().split(' ')

words = [word for word in words if (word not in stopwords and len(word) > 2)]

top_words = FreqDist(words).most_common(10)

top_words
# Get only words from top_words

words = []

for word, _ in top_words:

    words.append(word)



has_popular_words = train['original_title'].apply(lambda title: len([True for word in words if (word in title.lower())]))

(pd.concat([train['revenue'], has_popular_words], axis=1).groupby('original_title')['revenue'].mean()

 .plot(kind='bar', title='Popular words in title vs revenue'))
text = ' '.join(train['overview'].astype(str))



wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)



plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
train['popularity'].describe()
train[['original_title','popularity']].sort_values(by='popularity', ascending=False).head(10)
sns.jointplot(x='budget', y='popularity', data=train, kind='reg')
sns.lmplot(x='popularity', y='revenue', data=train)
sns.lmplot(x='popularity', y='profit', data=train)
# Drop poster_path feature

#train.drop(labels=['poster_path'], axis=1, inplace=True)

#test.drop(labels=['poster_path'], axis=1, inplace=True)
train['production_companies'] = train['production_companies'].apply(parse_json)

train['production_companies'].apply(len).value_counts().sort_index().plot(kind='bar')
# Get only list of names from list of directories

def parse_to_names(l):

    return [str(x['name']) for x in l]



companies = Counter(','.join(train['production_companies'].apply(parse_to_names)

                             .apply(lambda x: ','.join(x))).split(',')).most_common(21)

c = []

v = []

for x in companies[1:-1]:

    c.append(x[0])

    v.append(x[1])



df = pd.DataFrame(data={'company':c, 'count':v})



g = sns.barplot(x='company', y='count', data=df).set_xticklabels(rotation=90, labels=df['company'])
# Get full list of unique values from selected json feature

def get_list(df, feature):

    l = []

    for row in df[feature].apply(parse_to_names):

        l.extend(x for x in row if x not in l)

    return l



# Get companies names

companies = get_list(train, 'production_companies')



# Show five top entries

companies[:5]
# One hot encode all elements from list l with all movies based on json feature

# Returns new dataframe with one hot encoded data

def list_one_hot_encoding(df, feature, l):

    new_df = df.copy()

    for x in l:

        new_df[x] = df[feature].apply(parse_to_names).apply(lambda y: 1 if x in y else 0)

    return new_df

            

# One hot encoding of prodction companies

companies_df = list_one_hot_encoding(train, 'production_companies', companies)
# Show top rows of encoded production companies

companies_df[companies[:5]].head()
# Create new dataframe with aggregated data for passed list 'l' based on 'df' dataframe.

# name - name of elements in list

# Return new dataframe(name, movies_produced, most_popular_genre1, genre_1_count, top_language, language_count,

#                      mean_popularity, mean_budget, mean_revenue, mean_profit)

def aggregate_data(df, l, name):

    aggregated_df = pd.DataFrame(columns=[name, 'movies_produced', 'most_popular_genre1', 'genre1_count',

                                          'top_language', 'language_count', 'mean_popularity', 'mean_budget', 

                                          'mean_revenue', 'mean_profit'])

    for x in l:

        # Group data by element from the list and get its group

        group_df = df.groupby(x).get_group(1)

        

        # Create new row data and append it to dataframe

        aggregated_df = aggregated_df.append({name: x,

                                              'movies_produced': group_df['id'].count(),

                                              'most_popular_genre1': group_df['genre1'].value_counts().index[0],

                                              'genre1_count': group_df['genre1'].value_counts()[0],

                                              'top_language': group_df['original_language'].value_counts().index[0],

                                              'language_count': group_df['original_language'].value_counts()[0],

                                              'mean_popularity': group_df['popularity'].mean(),

                                              'mean_budget': group_df['budget'].mean(),

                                              'mean_revenue': group_df['revenue'].mean(),

                                              'mean_profit': group_df['profit'].mean()},

                                             ignore_index=True)

        

    # Return dataframe with aggregate data

    return aggregated_df

        

# Get aggregate companies data

companies_df = aggregate_data(companies_df, companies, 'company')



# Show top rows of companies dataframe

companies_df.head()
# Plot with number of movies produced by each company was done earlier by different method.

# Which company produced the most movies?

#sns.barplot(x='company', y='movies_produced', data=companies_df.sort_values(by='movies_produced', ascending=False).head(20))
g = sns.catplot(x='company', y='genre1_count', hue='most_popular_genre1', 

                data=companies_df.sort_values(by='movies_produced', ascending=False).head(20), aspect=3)

g.set_xticklabels(rotation=90)
plt.figure(figsize=(15,5))

g = (sns.barplot(x='company', y='mean_popularity', 

                data=companies_df.sort_values(by='mean_popularity', ascending=False).head(20))

     .set_xticklabels(rotation=90, labels=companies_df.sort_values(by='mean_popularity', ascending=False)

                      .head(20)['company']))
companies_df.sort_values(by='mean_popularity', ascending=False)[['company','movies_produced']].head(20)
tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_popularity', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='company', y='mean_popularity', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])
tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_budget', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='company', y='mean_budget', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])
tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_revenue', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='company', y='mean_revenue', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])
tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_profit', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='company', y='mean_profit', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])
# Delete companies dataframe from memory

del companies_df
# Show random values

train['production_countries'].sample(5)
train['production_countries'].apply(parse_json).apply(len).value_counts().plot(kind='bar')
# Create new feature 'number of production countries'

train['production_countries_count'] = train['production_countries'].apply(parse_json).apply(len)

test['production_countries_count'] = test['production_countries'].apply(parse_json).apply(len)
# Parse countries to json

train['production_countries'] = train['production_countries'].apply(parse_json)
# Get list of unique production countries

countries = get_list(train, 'production_countries')



# Get countries data

countries_df = aggregate_data(list_one_hot_encoding(train, 'production_countries', countries), countries, 'country')



# Show top rows of countries dataframe

countries_df.head(5)
len(countries)
tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_popularity', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='country', y='mean_popularity', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])
tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_budget', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='country', y='mean_budget', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])
tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_revenue', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='country', y='mean_revenue', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])
tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_profit', ascending=False).head(20)



plt.figure(figsize=(15,5))

g = sns.barplot(x='country', y='mean_profit', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])
# Delete coutries dataframe from memory

del countries_df
# Top 'release_date' rows

train['release_date'].head(10)
# Convert correctly year from YY to YYYY format.

def convert_date(date):

    date = date.split('/')

    if int(date[2]) < 19:

        date[2] = '20' + date[2]

    else:

        date[2] = '19' + date[2]

    return '/'.join(date)
# Convert year (when you only use pandas to_datatime function, you will get wrong dates)

train['release_date'] = train['release_date'].apply(convert_date)



# Convert data to daytime64 type

train['release_date'] = pd.to_datetime(train['release_date'])



# Show converted data

train['release_date'].head(5)
# Split data to parts as new features

train['release_date_year'] = train['release_date'].map(lambda x: x.year)

train['release_date_month'] = train['release_date'].map(lambda x: x.month)

train['release_date_day'] = train['release_date'].map(lambda x: x.day)

train['release_date_day_of_week'] = train['release_date'].apply(lambda x: x.dayofweek)
tmp = train['release_date_year'].value_counts()



plt.figure(figsize=(15,5))

ax = sns.lineplot(x=tmp.index, y=tmp.values)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
sns.countplot(x='release_date_month', data=train)
plt.figure(figsize=(10,5))

sns.countplot(x='release_date_day', data=train)
sns.countplot(x='release_date_day_of_week', data=train)
tmp = train.groupby(by='release_date_year')['popularity'].agg('mean')



plt.figure(figsize=(15,5))

ax = sns.lineplot(x=tmp.index, y=tmp.values)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
tmp = train.groupby(by='release_date_month')['popularity'].agg('mean')



plt.figure(figsize=(15,5))

ax = sns.barplot(x=tmp.index, y=tmp.values)
tmp = train.groupby(by='release_date_day')['popularity'].agg('mean')



plt.figure(figsize=(15,5))

ax = sns.barplot(x=tmp.index, y=tmp.values)
tmp = train.groupby(by='release_date_day_of_week')['popularity'].agg('mean')



plt.figure(figsize=(15,5))

ax = sns.barplot(x=tmp.index, y=tmp.values)
tmp = train.groupby(by='release_date_year')['budget'].agg('mean')



plt.figure(figsize=(15,5))

ax = sns.lineplot(x=tmp.index, y=tmp.values)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
tmp = train.groupby(by='release_date_month')['budget'].agg('mean')



plt.figure(figsize=(15,5))

sns.barplot(x=tmp.index, y=tmp.values)
fig, ax = plt.subplots(2, 1, figsize=(15,10))



tmp = train.groupby(by='release_date_day')['budget'].agg('mean')

sns.barplot(x=tmp.index, y=tmp.values, ax=ax[0])

ax[0].set_title('Average budget in case of day of releasing a movie')



tmp = train.groupby(by='release_date_day_of_week')['budget'].agg('mean')

sns.barplot(x=tmp.index, y=tmp.values, ax=ax[1])

ax[1].set_title('Average budget in case of releasing a movie in different day of the week')
plt.figure(figsize=(15,5))



tmp = train.groupby(by='release_date_year')['revenue'].agg('mean')

ax = sns.lineplot(x=tmp.index, y=tmp.values, label='Revenue')



tmp = train.groupby(by='release_date_year')['profit'].agg('mean')

ax = sns.lineplot(x=tmp.index, y=tmp.values, label='Profit')



ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
fix, ax = plt.subplots(2, 1, figsize=(15,10))



sns.set_color_codes('pastel')

tmp = train.groupby(by='release_date_month')['revenue'].agg('mean')

sns.barplot(x=tmp.index, y=tmp.values, ax=ax[0], label='Revenue', color='r')



sns.set_color_codes('muted')

tmp = train.groupby(by='release_date_month')['profit'].agg('mean')

sns.barplot(x=tmp.index, y=tmp.values, ax=ax[0], label='Profit', color='r')



ax[0].legend(loc="upper left")

ax[0].set_title('Revenue and profit difference over the months')



sns.set_color_codes('pastel')

tmp = train.groupby(by='release_date_day')['revenue'].agg('mean')

sns.barplot(x=tmp.index, y=tmp.values, ax=ax[1], label='Revenue', color='b')



sns.set_color_codes('muted')

tmp = train.groupby(by='release_date_day')['profit'].agg('mean')

sns.barplot(x=tmp.index, y=tmp.values, ax=ax[1], label='Profit', color='b')



ax[1].legend(loc="upper left")

ax[1].set_title('Revenue and profit difference over the days')
plt.figure(figsize=(15,5))



sns.set_color_codes('pastel')

tmp = train.groupby(by='release_date_day_of_week')['revenue'].agg('mean')

ax = sns.barplot(x=tmp.index, y=tmp.values, label='Revenue', color='r')



sns.set_color_codes('muted')

tmp = train.groupby(by='release_date_day_of_week')['profit'].agg('mean')

ax = sns.barplot(x=tmp.index, y=tmp.values, label='Profit', color='r')



ax.legend(loc="upper left")
train['runtime'].hist(bins=15)

print('Average runtime: ', train['runtime'].mean())

print('Std of runtime: ', train['runtime'].std())

print('Maximum runtime: ', train['runtime'].max())
plt.figure(figsize=(20, 6))

sns.lineplot(x='runtime', y='budget', data=train, label='budget')

sns.lineplot(x='runtime', y='revenue', data=train, label='revenue')

sns.lineplot(x='runtime', y='profit', data=train, label='profit').set_title('Impact of runtime to budget, revenue and profit')
plt.figure(figsize=(20,5))

tmp = train.groupby(by='genre1')['runtime'].mean()

sns.barplot(x=tmp.index, y=tmp.values)
# Show top rows of spoken_languages

train['spoken_languages'].head()
# Parse spoken_languages to json

train['spoken_languages'] = train['spoken_languages'].apply(parse_json)
train['spoken_languages'].apply(len).hist(bins=10)

print('Average: ', train['spoken_languages'].apply(len).mean())

print('Maximum languages used: ', train['spoken_languages'].apply(len).max())
# Create new feature representing number of languages spoken in a movie

train['spoken_languages_count'] = train['spoken_languages'].apply(len)
train[train['spoken_languages_count'] >= 6][['original_title','release_date','spoken_languages_count']].sort_values(by='spoken_languages_count')
sns.barplot(x='spoken_languages_count', y='budget', data=train)
plt.figure(figsize=(15,5))

sns.barplot(x='spoken_languages_count', y='revenue', data=train, label='revenue')

sns.lineplot(x='spoken_languages_count', y='profit', data=train, label='profit')
train.groupby(by='genre1')['spoken_languages_count'].mean()
# Get list of languages

languages = get_list(train, 'spoken_languages')



# Get languages data

languages_df = aggregate_data(list_one_hot_encoding(train, 'spoken_languages', languages), languages, 'language')



# Show top rows of languages dataframe

languages_df.head(5)
tmp = languages_df[languages_df['movies_produced'] > 20].sort_values(by='mean_popularity', ascending=False)



plt.figure(figsize=(20,5))

g = sns.barplot(x='language', y='mean_popularity', data=tmp).set_xticklabels(rotation=90, labels=tmp['language'].values)



tmp[['language','mean_popularity','movies_produced']]
tmp = languages_df[languages_df['movies_produced'] > 20].sort_values(by='mean_budget', ascending=False)



plt.figure(figsize=(20,5))

g = sns.barplot(x='language', y='mean_budget', data=tmp).set_xticklabels(rotation=90, labels=tmp['language'].values)



tmp[['language','mean_budget','movies_produced']]
tmp = languages_df[languages_df['movies_produced'] > 20].sort_values(by='mean_profit', ascending=False)



plt.figure(figsize=(20,5))



sns.set_color_codes('pastel')

ax = sns.barplot(x='language', y='mean_revenue', data=tmp, label='mean revenue', color='b')



sns.set_color_codes('muted')

ax = sns.barplot(x='language', y='mean_profit', data=tmp, label='mean profit', color='b')



ax.legend(loc='upper right')



tmp[['language','movies_produced','mean_revenue','mean_profit']]
# Print unique values for status

print('From train set: ', train['status'].unique().tolist())

print('From test set: ', test['status'].unique().tolist())
train[train['status'] == 'Rumored'][['original_title','release_date','revenue','status']]
test[test['status'] == 'Rumored'][['original_title','release_date','status']]
test[test['status'] == 'Post Production'][['original_title','release_date','status']]
test[test['status'].isnull()][['original_title','release_date','status']]
# Delete status feature

#train = train.drop(labels=['status'], axis=1)

#test = test.drop(labels=['status'], axis=1)
how_many_missing_values('tagline')
# Create new feature indicating if a movie has a tagline

train['has_tagline'] = train['tagline'].notnull()

test['has_tagline'] = test['tagline'].notnull()
# Print sample taglines

train[train['has_tagline']][['original_title','tagline']].sample(10)
# Get all taglines

text = ' '.join(train[train['has_tagline']]['tagline'])



# Generate WordCloud from titles

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)



# Display wordcloud

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
sns.boxplot(x='has_tagline', y='revenue', data=train)
print('Movies with different titles: ', train[train['title'] != train['original_title']].shape[0])



# Print sample movies with different titles

train[train['title'] != train['original_title']][['original_title', 'title']].sample(10)
# Create new feature indicating if a movie has different original title and title

train['has_different_titles'] = train['original_title'] != train['title']

test['has_different_titles'] = test['original_title'] != test['title']
sns.boxplot(x='has_different_titles', y='revenue', data=train)
how_many_missing_values('Keywords')
# Parse to json

train['Keywords'] = train['Keywords'].apply(parse_json)
print('Train set Keywords parameters:\n', train['Keywords'].apply(len).describe())

ax = train['Keywords'].apply(len).hist(bins=30)

ax = test['Keywords'].apply(parse_json).apply(len).hist(bins=30)



ax.legend(labels=['train','test'])
train[train['Keywords'].apply(len) > 100][['original_title','release_date','production_countries','Keywords']]
# Get all keywords

text = ','.join(train['Keywords'].apply(parse_to_names).apply(lambda x: ','.join(x)))



# Generate WordCloud from titles

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)



# Display wordcloud

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()

keywords = Counter(text.split(',')).most_common(201)[1:]



# Print keywords

keywords
# Get only keywords from most common keywords list

keywords = [x[0] for x in keywords]
# One hot encode most common keywords

keywords_df = list_one_hot_encoding(train, 'Keywords', keywords)
fig, ax = plt.subplots(40,5, figsize=(20,160))

for x in range(40):

    for y in range(5):

        sns.boxplot(x=keywords[(x*5)+y], y='revenue', data=keywords_df, ax=ax[x][y])
# From above list let's select keywords that I thought have some impact on revenue.

keywords = ['woman director',

            'independent film',

            'duringcreditsstinger',

            'based on novel',

            'violence',

            'dystopia',

            'aftercreditsstinger',

            'sequel',

            'los angeles',

            'new york',

            '3d',

            'based on comic',

            'escape',

            'alien',

            'london england',

            'superhero',

            'corruption',

            'martial arts',

            'dying and death',

            'brother sister relationship',

            'soldier',

            'future',

            'lawyer',

            'post-apocalyptic',

            'magic',

            'explosion',

            'time travel',

            'romantic comedy',

            'monster',

            'helicopter',

            'rescue',

            'assassin',

            'cia',

            'terrorist',

            'fight',

            'fbi',

            'scientist',

            'zombie',

            'psychopath',

            'survival',

            'competition',

            'spy',

            'found footage',

            'sister sister relationship',

            'motorcycle',

            'england',

            'conspiracy',

            'fbi agent',

            'island',

            'faith',

            'amnesia',

            'undercover',

            'baby',

            'marvel comic',

            'california',

            'assassination',

            'government',

            'military',

            'army',

            'mountain',

            'animation',

            'holiday',

            'comedy',

            'robot',

            'police officer',

            'parent child relationship',

            'car crash',

            'car chase',

            'jungle',

            'usa president',

            'fire',

            'secret identity',

            'based on young adult novel',

            'christmas',

            'washington d.c.',

            'super powers',

            'snow',

            'mutant',

            'restaurant',

            'hero',

            'dancing',

            'space',

            'airport'

           ]



# List is longer than I assume.
# Create from selected keywords new features

train = list_one_hot_encoding(train, 'Keywords', keywords)
# Remove keywords_df from memory

del keywords_df
# Parse cast data.

train['cast'] = train['cast'].replace('\'','\"').apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
print(train['cast'].apply(len).describe())

train['cast'].apply(len).hist(bins=15)
(train[train['cast'].apply(len) >= 80][['original_title','release_date','revenue','cast']]

 .assign(cast_count=lambda x: x['cast'].apply(len)).sort_values(by='cast_count'))
# Get all character names

text = ','.join(train['cast'].apply(lambda x: [actor['character'] for actor in x]).apply(lambda x: ','.join(x)))



# Generate WordCloud from characters names

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)



# Display wordcloud

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
Counter(text.split(',')).most_common(25)
# Count gender values form 'df' dataframe from 'feature' column.

# Returns Series with gender value counts.

def gender_counts(df, feature):

    gender_count = pd.Series(data=[0,0,0], index=[0,1,2])

    for row in df[feature]:

        for person in row:

            gender_count.loc[person['gender']] += 1

    return gender_count
tmp = gender_counts(train, 'cast')

print(tmp)



sns.barplot(x=tmp.index, y=tmp.values)
# Create new dataframe with aggregated actors data.

# Returns new dataframe with columns:

#  [id, name, movies_played, mean_gender, mean_order, start_year, end_year, 

#   mean_movie_popularity, mean_movie_budget, mean_movie_revenue, mean_movie_profit]

def aggregate_actors_data(df):

    

    #Create new dataframe for actors data

    actors_df = pd.DataFrame(columns=['id','name','movies_played','mean_gender','mean_order',

                                      'start_year','end_year','mean_movie_popularity','mean_movie_budget',

                                      'mean_movie_revenue','mean_movie_profit'])

    

    # Iterate through movies

    for index, movie in df.iterrows():

        

        #Iterate through cast list

        for actor in movie['cast']:

            

            #Chack if actor is already in dataframe

            if actors_df['id'].isin([actor['id']]).sum() == 0:

                

                # If not create new entry for actor

                actors_df = actors_df.append({'id': actor['id'],

                                              'name': actor['name'],

                                              'movies_played': 1,

                                              'mean_gender': actor['gender'],

                                              'mean_order': actor['order'],

                                              'start_year': movie['release_date_year'],

                                              'end_year': movie['release_date_year'],

                                              'mean_movie_popularity': movie['popularity'],

                                              'mean_movie_budget': movie['budget'],

                                              'mean_movie_revenue': movie['revenue'],

                                              'mean_movie_profit': movie['profit']

                                             },

                                             ignore_index=True)

            else:

                # If exists, then update values

                actors_df.loc[actors_df['id'] == actor['id'], 'movies_played'] += 1

                actors_df.loc[actors_df['id'] == actor['id'], 'mean_gender'] += actor['gender']

                actors_df.loc[actors_df['id'] == actor['id'], 'mean_order'] += actor['order']

                actors_df.loc[actors_df['id'] == actor['id'], 'start_year'] = min(actors_df.loc[actors_df['id'] == actor['id'], 'start_year'].values[0], 

                                                                                  movie['release_date_year'])

                actors_df.loc[actors_df['id'] == actor['id'], 'end_year'] = max(actors_df.loc[actors_df['id'] == actor['id'], 'end_year'].values[0], 

                                                                                  movie['release_date_year'])

                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_popularity'] += movie['popularity']

                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_budget'] += movie['budget']

                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_revenue'] += movie['revenue']

                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_profit'] += movie['profit']



    # Get mean values

    actors_df['mean_gender'] = actors_df['mean_gender'] / actors_df['movies_played']

    actors_df['mean_order'] = actors_df['mean_order'] / actors_df['movies_played']

    actors_df['mean_movie_popularity'] = actors_df['mean_movie_popularity'] / actors_df['movies_played']

    actors_df['mean_movie_budget'] = actors_df['mean_movie_budget'] / actors_df['movies_played']

    actors_df['mean_movie_revenue'] = actors_df['mean_movie_revenue'] / actors_df['movies_played']

    actors_df['mean_movie_profit'] = actors_df['mean_movie_profit'] / actors_df['movies_played']

    

    # Change data types to numeric ones.

    actors_df['id'] = actors_df['id'].astype('int64')

    actors_df['movies_played'] = actors_df['movies_played'].astype('int64')

    actors_df['mean_gender'] = actors_df['mean_gender'].astype('float64')

    actors_df['mean_order'] = actors_df['mean_order'].astype('float64')

    actors_df['mean_movie_popularity'] = actors_df['mean_movie_popularity'].astype('float64')

    actors_df['mean_movie_budget'] = actors_df['mean_movie_budget'].astype('float64')

    actors_df['mean_movie_revenue'] = actors_df['mean_movie_revenue'].astype('float64')

    actors_df['mean_movie_profit'] = actors_df['mean_movie_profit'].astype('float64')

    

    # Return new dataframe

    return actors_df
# Get aggregated actors data

#actors_df = aggregate_actors_data(train)



# Save actors dataframe to file

#actors_df.to_csv('../actors.csv', index=False)



# Load actors dataframe from file

actors_df = pd.read_csv('../input/tmdb-box-office-prediction-cast-crew/actors.csv')



# Show sample actors data

actors_df.sample(10)
# Get all actors names

text = ','.join(actors_df['name'])



# Generate WordCloud from actors names

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)



# Display wordcloud

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
print('Actors number: ', actors_df.shape[0])
print('Number of actors that played in only one movie: ', actors_df[actors_df['movies_played'] == 1].shape[0])
# Lets drop all actors that played in only one movie.

# (Those actors have no meaningful informations. Would only obscure the real picture)

actors_df = actors_df.drop(actors_df[actors_df['movies_played'] == 1].index, axis=0)
g = sns.barplot(y='name', x='movies_played', hue='mean_gender', 

                data=actors_df.sort_values(by='movies_played', ascending=False).head(20), palette='rocket')
actors_df['mean_gender'].value_counts()
actors_df.groupby(by='mean_gender')['movies_played'].mean().plot(kind='pie', legend=False, autopct='%1.0f%%', 

                                                                 labels=['unknown','female','male'])
sns.barplot(x='mean_gender', y='mean_order', data=actors_df)
tmp = actors_df[actors_df['movies_played'] > 7].sort_values(by='mean_order').head(25)



plt.figure(figsize=(10,10))

sns.barplot(x='mean_order', y='name', data=tmp)
# Calculate years of professional activity

actors_df = actors_df.assign(years_active=lambda x: x['end_year'] - x['start_year'])



plt.figure(figsize=(10,10))

tmp = actors_df[actors_df['movies_played'] > 7].sort_values(by='years_active', ascending=False).head(25)



sns.barplot(x='years_active', y='name', data=tmp)
sns.barplot(x='mean_movie_popularity', y='name', data=actors_df.sort_values(by='mean_movie_popularity', 

                                                                            ascending=False).head(25))
actors_df.sort_values(by='mean_movie_popularity',ascending=False).head(25)[['name','movies_played']]
tmp = actors_df[actors_df['movies_played'] >= 10].sort_values(by='mean_movie_popularity', ascending=False).head(40)

plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_popularity', y='name', data=tmp)
tmp = actors_df[actors_df['movies_played'] > 8].sort_values(by='mean_movie_revenue', ascending=False).head(50)



traces = []

for _, row in tmp.iterrows():

    traces.append(go.Scatter(

        x=[row['mean_movie_budget']],

        y=[row['mean_movie_revenue']],

        name=row['name'],

        mode='markers'

    ))



layout = go.Layout(

    title = go.layout.Title(text='Actors based on mean movies revenue and budget'),

    xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(text='Mean movie budget')), 

    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text='Mean movie revenue'))

)



fig = go.Figure(data=traces, layout=layout)

iplot(fig, show_link=False)



# Seaborn version (no interactive)

#plt.figure(figsize=(16,10))

#sns.stripplot(x='mean_movie_budget', y='mean_movie_revenue', hue='name', data=tmp)
tmp = actors_df[actors_df['movies_played'] > 7].sort_values(by='mean_movie_profit', ascending=False).head(25)



plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_profit', y='name', data=tmp)
# Get IDs for top 100 actors that have biggest mean movie revenue.

# Excluding all actros that played in less than nine movies.

ids = (actors_df[actors_df['movies_played'] > 8]

       .sort_values(by='mean_movie_revenue', ascending=False)['id'].head(100).values.tolist())
# Check if person worked in a movie.

def is_person_in_cast(cast, ids):

    for person in cast:

        if person['id'] in ids:

            return True

    return False
# Create new feature indicating if in a movie played any of actors from the above list.

train['has_top_actor'] = train['cast'].apply(lambda x: is_person_in_cast(x, ids))
# Get top 6 actors

def top_actors(cast):

    actors = pd.Series(data=['', '', '', '', '', ''],

                       index=['actor0', 'actor1', 'actor2', 'actor3', 'actor4', 'actor5'])

    for n in range(min(6, len(cast))):

        actors.loc['actor{}'.format(n)] = cast[n]['name']

    return actors
# Create new feature for top 6 actors in the cast

train[['actor0', 'actor1', 'actor2', 'actor3', 'actor4', 'actor5']] = train['cast'].apply(top_actors)
# Remove actors dataframe to free memory

del actors_df
tmp = train.groupby(by='has_top_actor')['revenue'].mean()

sns.barplot(x=tmp.index, y=tmp.values)



print('How numerous each group are:\n', train['has_top_actor'].value_counts())
# Parse crew data

train['crew'] = train['crew'].replace('\'','\"').apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
# Show histogram

train['cast'].apply(len).hist()



# Describe data

train['cast'].apply(len).describe()
gender = gender_counts(train, 'crew')



print('Gender counts:\n', gender)



sns.barplot(x=gender.index, y=gender.values)
# Return list of departments

def get_departments(df):

    departments = set()

    for crew in df:

        for person in crew:

            departments.add(person['department'])

    return list(departments)
# Return list of jobs by departments

def get_jobs_by_department(df, departments):

    jobs = {department:set() for department in departments}

    for crew in df:

        for person in crew:

            jobs[person['department']].add(person['job'])

    return jobs
# Get list of departments

departments = get_departments(train['crew'])



# Get list of jobs by departments

jobs = get_jobs_by_department(train['crew'], departments)



# Print jobs

jobs
['{}: {}'.format(x, len(jobs[x])) for x in jobs]
# I manually selected below jobs (by my intuition)

selected_jobs =  ['Casting', 'Original Music Composer', 'Screenplay', 'Director', 'Director of Photography', 

                  'Editor', 'Costume Design', 'Producer', 'Executive Producer']
# Get list of person jobs from crew data

def get_person_jobs(df, person_id):

    person_jobs = set()

    for crew in df:

        for person in crew:

            if person['id'] == person_id:

                person_jobs.add(person['job'])

    return list(person_jobs)
# Aggregate crew data and return new dataframe with features:

# [id, name, gender, job, movies_created, start_year, end_year, 

#  mean_movie_popularity, mean_movie_budget, mean_movie_revenue, mean_movie_profit]

def aggregate_crew_data(df, jobs):

    

    # Create new dataframe

    crew_df = pd.DataFrame(columns=['id', 'name', 'gender', 'job', 'movies_created', 'start_year', 'end_year',

                                    'mean_movie_popularity', 'mean_movie_budget','mean_movie_revenue',

                                    'mean_movie_profit'])

    

    # Iterate through movies

    for index, movie in df.iterrows():

        

        # Iterate through crew members

        for person in movie['crew']:

            

            # Check if person works in one of selected jobs

            if person['job'] in jobs:

            

                # Chack if person is already in dataframe

                if crew_df['id'].isin([person['id']]).sum() == 0:



                    # If not, then add new entry for crew member

                    crew_df = crew_df.append({'id': person['id'],

                                              'name': person['name'],

                                              'gender': person['gender'],

                                              'job': get_person_jobs(df['crew'], person['id']),

                                              'movies_created': 1,

                                              'start_year': movie['release_date_year'],

                                              'end_year': movie['release_date_year'],

                                              'mean_movie_popularity': movie['popularity'],

                                              'mean_movie_budget': movie['budget'],

                                              'mean_movie_revenue': movie['revenue'],

                                              'mean_movie_profit': movie['profit']

                                             },

                                             ignore_index=True)



                else:



                    # If exists, then update values

                    crew_df.loc[crew_df['id'] == person['id'], 'movies_created'] += 1

                    crew_df.loc[crew_df['id'] == person['id'], 'start_year'] = min(crew_df.loc[crew_df['id'] == person['id'], 'start_year'].values[0],

                                                                                   movie['release_date_year'])

                    crew_df.loc[crew_df['id'] == person['id'], 'end_year'] = max(crew_df.loc[crew_df['id'] == person['id'], 'end_year'].values[0],

                                                                                   movie['release_date_year'])

                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_popularity'] += movie['popularity']

                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_budget'] += movie['budget']

                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_revenue'] += movie['revenue']

                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_profit'] += movie['profit']

            

    # Get mean values

    crew_df['mean_movie_popularity'] = crew_df['mean_movie_popularity'] / crew_df['movies_created']

    crew_df['mean_movie_budget'] = crew_df['mean_movie_budget'] / crew_df['movies_created']

    crew_df['mean_movie_revenue'] = crew_df['mean_movie_revenue'] / crew_df['movies_created']

    crew_df['mean_movie_profit'] = crew_df['mean_movie_profit'] / crew_df['movies_created']

    

    # Change data types to numeric ones

    crew_df['id'] = crew_df['id'].astype('int64')

    crew_df['gender'] = crew_df['gender'].astype('int64')

    crew_df['movies_created'] = crew_df['movies_created'].astype('int64')

    crew_df['mean_movie_popularity'] = crew_df['mean_movie_popularity'].astype('float64')

    crew_df['mean_movie_budget'] = crew_df['mean_movie_budget'].astype('float64')

    crew_df['mean_movie_revenue'] = crew_df['mean_movie_revenue'].astype('float64')

    crew_df['mean_movie_profit'] = crew_df['mean_movie_profit'].astype('float64')

    

    # Return crew dataframe

    return crew_df
# Get aggregated crew data

#crew_df = aggregate_crew_data(train, selected_jobs)



# Save crew dataframe to file

#crew_df.to_csv('../crew.csv', index=False)



# Load crew dataframe from file

crew_df = pd.read_csv('../input/tmdb-box-office-prediction-cast-crew/crew.csv')



# Show sample crew data

crew_df.sample(10)
# Remove all crew members with only one movie created and with popularity less than 9.

crew_df = crew_df[~((crew_df['movies_created'] == 1) & (crew_df['mean_movie_popularity'] < 9))]
tmp = crew_df[['name','job','movies_created']].sort_values('movies_created', ascending=False).head(25)

tmp['job'] = tmp['job'].apply(lambda x: x[0])



fig, ax = plt.subplots(1,2, figsize=(16,10), gridspec_kw = {'width_ratios':[1, 2]})

sns.barplot(x='movies_created', y='name', data=tmp, ax=ax[0])



ax[1].axis('off')

table = ax[1].table(cellText=tmp.values, 

                  rowLabels=tmp.index, 

                  colLabels=tmp.columns, 

                  bbox=[0,0,1,1])

table.auto_set_font_size(False)

table.set_fontsize(14)
crew_df = crew_df.assign(years_active=lambda x: x['end_year'] - x['start_year'])



tmp = crew_df[['name','start_year','end_year','years_active']].sort_values(by='years_active', ascending=False).head(25)



fig, ax = plt.subplots(1,2, figsize=(16,10), gridspec_kw = {'width_ratios':[1, 2]})

sns.barplot(x='years_active', y='name', data=tmp, ax=ax[0])



ax[1].axis('off')

table = ax[1].table(cellText=tmp.values, 

                  rowLabels=tmp.index, 

                  colLabels=tmp.columns, 

                  bbox=[0,0,1,1])

table.auto_set_font_size(False)

table.set_fontsize(14)
tmp = crew_df[crew_df['movies_created'] > 7].sort_values('mean_movie_popularity', ascending=False).head(50)



plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_popularity', y='name', data=tmp)
# Get the list of 'popular' crew members IDs

crew_popularity_ids = tmp['id'].values.tolist()
# Create new feature indicating if a movie was created by one of 'popular' crew members

train['has_popular_crew'] = train['crew'].apply(lambda x: is_person_in_cast(x, crew_popularity_ids))
tmp = crew_df[crew_df['movies_created'] > 7].sort_values(by='mean_movie_revenue', ascending=False).head(50)



traces = []

for _, row in tmp.iterrows():

    traces.append(go.Scatter(

        x=[row['mean_movie_budget']],

        y=[row['mean_movie_revenue']],

        name=row['name'],

        mode='markers'

    ))



layout = go.Layout(

    title = go.layout.Title(text='Crew members based on mean movies revenue and budget'),

    xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(text='Mean movie budget')), 

    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text='Mean movie revenue'))

)



fig = go.Figure(data=traces, layout=layout)

iplot(fig, show_link=False)
# Get IDs for top 100 crew members that have biggest mean movie revenue.

# Excluding all crew members that worked on less than eight movies.

crew_revenue_ids = crew_df[crew_df['movies_created'] > 7].sort_values(by='mean_movie_revenue', ascending=False).head(100)
# Create new feature indicating if a movie was created by one of crew members with biggest mean revenue

train['has_top_crew'] = train['crew'].apply(lambda x: is_person_in_cast(x, crew_revenue_ids))
tmp = crew_df[crew_df['movies_created'] > 7].sort_values(by='mean_movie_profit', ascending=False).head(50)



plt.figure(figsize=(15,10))



sns.set_color_codes('pastel')

ax = sns.barplot(x='mean_movie_revenue', y='name', data=tmp, label='mean revenue', color='b')



sns.set_color_codes('muted')

ax = sns.barplot(x='mean_movie_profit', y='name', data=tmp, label='mean profit', color='b')



ax.legend(loc='lower right')
fig = plt.figure(figsize=(15,30))



for x, job in enumerate(selected_jobs):

    tmp = (crew_df[(crew_df['job'].apply(lambda x: job in x)) & (crew_df['movies_created'] > 7)]

           .sort_values('mean_movie_revenue', ascending=False).head(5))



    ax = fig.add_subplot(5,2,x+1)

    sns.barplot(x='mean_movie_revenue', y='name', data=tmp, ax=ax)

    ax.set_title(job)
tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Director' in x)) & (crew_df['movies_created'] > 7)]

       .sort_values('mean_movie_revenue', ascending=False).head(25))



plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_revenue', y='name', data=tmp)
tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Casting' in x)) & (crew_df['movies_created'] > 7)]

       .sort_values('mean_movie_revenue', ascending=False).head(25))



plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_revenue', y='name', data=tmp)
tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Director of Photography' in x)) & (crew_df['movies_created'] > 7)]

       .sort_values('mean_movie_revenue', ascending=False).head(25))



plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_revenue', y='name', data=tmp)
tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Costume Design' in x)) & (crew_df['movies_created'] > 7)]

       .sort_values('mean_movie_revenue', ascending=False).head(25))



plt.figure(figsize=(10,10))

sns.barplot(x='mean_movie_revenue', y='name', data=tmp)
# Get person name of job role for the movie

def get_crew_role_name(crew, job):

    for person in crew:

        if person['job'] == job:

            return pd.Series(data=[person['name']], index=[job])

    return pd.Series(data=[''], index=[job])
# Create new feature for director

train['director'] = train['crew'].apply(lambda x: get_crew_role_name(x,'Director'))
directors = train.groupby('director')['revenue'].mean().sort_values(ascending=False).head(50)



plt.figure(figsize=(10, 10))

sns.barplot(x=directors.values, y=directors.index)
# Remove crew dataframe from memory

del crew_df