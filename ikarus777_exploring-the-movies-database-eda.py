import warnings

warnings.filterwarnings('ignore',category=FutureWarning)



import re

from os import path, getcwd

from datetime import datetime



import numpy as np

import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error



from PIL import Image

from wordcloud import WordCloud, STOPWORDS



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from IPython.display import display, HTML



pd.set_option('display.max_columns', 100)

plt.style.use('bmh')



ID = 'id'

TARGET = 'revenue'

NFOLDS = 5

SEED = 126

NROWS = None

DATA_DIR = '../input'



TRAIN_FILE = f'{DATA_DIR}/train.csv'

TEST_FILE = f'{DATA_DIR}/test.csv'
train_data = pd.read_csv(TRAIN_FILE, nrows=NROWS)

test_data = pd.read_csv(TEST_FILE, nrows=NROWS)
series_cols = ['belongs_to_collection', 'genres', 'production_companies',

               'production_countries', 'spoken_languages', 'Keywords',

               'cast', 'crew']

train = train_data.copy()

test = test_data.copy()

for df in [train, test]:

    for column in series_cols:

        df[column] = df[column].apply(lambda s: [] if pd.isnull(s) else eval(s))



full = pd.concat([train, test], sort=False)
def uniqueValues(df, col, key):

    all_values = []

    for record in df[col]:

        lst = [d[key] for d in record]

        all_values.extend(lst)

    all_values = np.array(all_values)

    unique, counts = np.unique(all_values, return_counts=True)

    return pd.DataFrame({ 'Value': unique, 'Counts': counts })



genres_unique = uniqueValues(full, 'genres', 'name').sort_values(by='Counts', ascending=False)

languages_unique = uniqueValues(full, 'spoken_languages', 'iso_639_1').sort_values(by='Counts', ascending=False)

top_languages = languages_unique.iloc[:4]



test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/00'
def fixYear(row):

    year = int(row.split('/')[2])

    return row[:-2] + str(year + (2000 if year <= 19 else 1900))



def extractField(row, value):

    if row is np.nan: return 0

    return 1 if value in row else 0



for df in [train, test]:

    df['genres_list'] = df['genres'].apply(lambda row: ','.join(d['name'] for d in row))

    df['genres_count'] = df['genres'].apply(lambda x: len(x))



    df['budget_to_popularity'] = df['budget'] / df['popularity']

    df['budget_to_runtime'] = df['budget'] / df['runtime']



    df['prod_companies_list'] = df['production_companies'].apply(lambda row: ','.join(d['name'] for d in row))

    df['prod_countries_list'] = df['production_countries'].apply(lambda row: ','.join(d['iso_3166_1'] for d in row))



    df['languages_list'] = df['spoken_languages'].apply(lambda row: ','.join(d['iso_639_1'] for d in row))



    for l in top_languages['Value'].values:

        df['lang_' + l] = df['languages_list'].apply(extractField, args=(l,))



    df['has_homepage'] = df['homepage'].apply(lambda v: pd.isnull(v) == False)



    df['release_date'] = df['release_date'].apply(fixYear)

    df['release_date'] = pd.to_datetime(df['release_date'])



    date_parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + '_' + part

        df[part_col] = getattr(df['release_date'].dt, part).astype(int)



    df['collection'] = df['belongs_to_collection'].apply(lambda row: ','.join(d['name'] for d in row))

    df['has_collection'] = df['collection'].apply(lambda v: 1 if v else 0)
train.sample(2).T
full = pd.concat([train, test], sort=False)
fig = plt.figure(figsize = (20, 6))

plt.subplot(1, 2, 1)

sns.distplot(train['revenue'])

plt.subplot(1, 2, 2)

sns.distplot(np.log1p(train['revenue']))

fig.suptitle('Revenue', fontsize=20)

plt.show()
plt.figure(figsize=(8, 8))

plt.scatter(train['release_date_year'], train['revenue'])

plt.title('Movie revenue per year')

plt.xlabel('Year')

plt.ylabel('Revenue')

plt.show()
top_movies = train.sort_values(by='revenue', ascending=False)

top_movies.head(10)[['title', 'revenue']]
train['profit'] = train.apply(lambda row: row['revenue'] - row['budget'], axis=1)
plt.figure(figsize = (16, 6))

sns.distplot(train['profit'])

plt.title('Profit')

plt.show()
sns.lmplot('revenue', 'budget', data=train)

plt.show()
worst_movies = train.sort_values(by='profit', ascending=False)

worst_movies.head(10)[['title', 'profit', 'budget', 'revenue']]
worst_movies = train.sort_values(by='profit', ascending=True)

worst_movies.head(10)[['title', 'profit', 'budget', 'revenue']]
plt.figure(figsize=(8, 8))

dataTrain = train['release_date_year'].value_counts().sort_index()

dataTest = test['release_date_year'].value_counts().sort_index()

plt.plot(dataTrain.index, dataTrain.values, label='train')

plt.plot(dataTest.index, dataTest.values, label='test')

plt.title('Number of movies per year')

plt.xlabel('Year')

plt.ylabel('Revenue')

plt.legend(loc='upper center', frameon=False)

plt.show()
fig = plt.figure(figsize = (20, 6))

plt.subplot(1, 2, 1)

sns.distplot(full['popularity'])

plt.subplot(1, 2, 2)

sns.distplot(np.log1p(full['popularity']))

fig.suptitle('Popularity (full)', fontsize=20)

plt.show()
plt.figure(figsize=(20, 8))



plt.subplot(1, 2, 1)

plt.scatter(full['popularity'], full['revenue'])

plt.title('Popularity vs revenue (full)')

plt.xlabel('Popularity')

plt.ylabel('Revenue')



plt.subplot(1, 2, 2)

plt.scatter(np.log1p(full['popularity']), np.log1p(full['revenue']))

plt.title('Popularity vs revenue - log(x + 1) (full)')

plt.xlabel('Popularity')

plt.ylabel('Revenue')



plt.show()
plt.figure(figsize=(8, 8))

plt.scatter(full['release_date_year'], full['popularity'])

plt.title('Popularity per year (full)')

plt.xlabel('Year')

plt.ylabel('Popularity')

plt.show()
fig = plt.figure(figsize = (20, 6))

plt.subplot(1, 2, 1)

sns.distplot(full['budget'])

plt.subplot(1, 2, 2)

sns.distplot(np.log1p(full['budget']))

fig.suptitle('Budget (full)', fontsize=20)

plt.show()
plt.figure(figsize=(20, 8))



plt.subplot(1, 2, 1)

plt.scatter(train['budget'], train['revenue'])

plt.title('Budget vs revenue')

plt.xlabel('Budget')

plt.ylabel('Revenue')



plt.subplot(1, 2, 2)

plt.scatter(np.log1p(train['budget']), np.log1p(train['revenue']))

plt.title('Budget vs revenue - log(x + 1)')

plt.xlabel('Budget')

plt.ylabel('Revenue')



plt.show()
fig = plt.figure(figsize = (20, 6))

full['runtime'].fillna(full['runtime'].mean(), inplace=True)

sns.distplot(full['runtime'])

fig.suptitle('Runtime (full)', fontsize=20)

plt.show()
plt.figure(figsize=(16, 8))

plt.bar(train['release_date_year'], train['revenue'], label='revenue')

plt.bar(train['release_date_year'], train['budget'], label='budget')

plt.title('Revenue/Budget per year')

plt.xlabel('Year')

plt.ylabel('Budget')

plt.legend(loc='upper center', frameon=False)

plt.show()
plt.figure(figsize=(12, 8))

plt.bar(train['release_date_month'], train['revenue'], label='revenue')

plt.bar(train['release_date_month'], train['budget'], label='budget')

plt.title('Revenue/Budget per Month')

plt.xlabel('Month')

plt.ylabel('Revenue / Budget')

plt.legend(loc='upper center', frameon=False)

plt.show()
plt.figure(figsize=(20, 8))

plt.bar(train['release_date_year'], train['popularity'], alpha=0.5)

plt.bar(test['release_date_year'], test['popularity'], alpha=0.5)

plt.title('Popularity per year')

plt.xlabel('Popularity (full)')

plt.ylabel('Year')

plt.show()
plt.figure(figsize=(16, 8))

ax = sns.barplot(x='Counts', y='Value', data=genres_unique, palette='Spectral')

ax.set_title(label='Distribution of genres')

ax.set_ylabel('')

ax.set_xlabel('Number of movies')

plt.show()
companies_unique = uniqueValues(full, 'production_companies', 'name').sort_values(by='Counts', ascending=False)



TOP_COMPANIES = 15

plt.figure(figsize=(16, 8))

ax = sns.barplot(x='Counts', y='Value', data=companies_unique[:TOP_COMPANIES], palette='Spectral')

ax.set_title(label='Distribution of top {} companies'.format(TOP_COMPANIES))

ax.set_ylabel('')

ax.set_xlabel('Number of movies')

plt.show()
prodc_unique = uniqueValues(full, 'production_countries', 'iso_3166_1').sort_values(by='Counts', ascending=False)



TOP_COUNTRIES = 15

plt.figure(figsize=(12, 6))

ax = sns.barplot(y='Counts', x='Value', data=prodc_unique[:TOP_COUNTRIES], palette='hot')

ax.set_title(label='Distribution of top {} production countries'.format(TOP_COUNTRIES))

ax.set_ylabel('')

ax.set_xlabel('')

plt.show()
TOP_LANGUAGES = 15

plt.figure(figsize=(12, 6))

ax = sns.barplot(y='Counts', x='Value', data=languages_unique[:TOP_LANGUAGES], palette='hot')

ax.set_title(label='Distribution of top {} languages'.format(TOP_LANGUAGES))

ax.set_ylabel('')

ax.set_xlabel('')

plt.show()
cast_unique = uniqueValues(full, 'cast', 'name').sort_values(by='Counts', ascending=False)



TOP_CAST = 25

plt.figure(figsize=(16, 8))

ax = sns.barplot(x='Counts', y='Value', data=cast_unique[:TOP_CAST], palette='BuPu_r')

ax.set_title(label='Distribution of top {} actors'.format(TOP_CAST))

ax.set_ylabel('')

ax.set_xlabel('Number of movies')

plt.show()
cast_unique = uniqueValues(full, 'cast', 'gender')



colors = [ '#F2B134', '#068587', '#ED553B']

labels=['Gender 0', 'Gender 1', 'Gender 2']

fig, ax = plt.subplots(figsize=(8, 6))

ax.pie(cast_unique['Counts'], labels=labels, colors=colors, autopct='%1.1f%%')

ax.axis('equal')

ax.set_title(label='Distribution of genders in actors')

plt.show()
keywords_unique = uniqueValues(full, 'Keywords', 'name').sort_values(by='Counts', ascending=False)



TOP_COMPANIES = 25

plt.figure(figsize=(16, 8))

ax = sns.barplot(x='Counts', y='Value', data=keywords_unique[:TOP_COMPANIES], palette='icefire_r')

ax.set_title(label='Most used Keywords')

ax.set_ylabel('')

ax.set_xlabel('')

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(train['overview'].fillna('').values)

wordcloud = WordCloud(margin=10, background_color='white', colormap='Greens', width=1200, height=1000).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Top words in overview', fontsize=20)

plt.axis('off')

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(train['title'].fillna('').values)

wordcloud = WordCloud(margin=10, background_color='white', colormap='Reds', width=1200, height=1000).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Top words in titles', fontsize=20)

plt.axis('off')

plt.show()
def getNullCols(df):

    total_null = df.isnull().sum().sort_values(ascending=False)

    percent_null = ((df.isnull().sum() / df.isnull().count()) * 100).sort_values(ascending=False)

    missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])

    return missing_data

null_df = getNullCols(train_data).head(10)



plt.figure(figsize=(10, 5))

sns.barplot(y=null_df.index, x=null_df['Total'], palette='icefire_r')

plt.title('Total null values by feature')

plt.xlabel('')

plt.ylabel('')

plt.show()

display(null_df)