import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

from scipy import stats



import warnings

warnings.filterwarnings('ignore')
books = pd.read_csv('../input/goodbooks-10k/books.csv')

ratings = pd.read_csv('../input/goodbooks-10k/ratings.csv')

book_tags = pd.read_csv('../input/goodbooks-10k/book_tags.csv')

tags = pd.read_csv('../input/goodbooks-10k/tags.csv')
books.head().transpose()
ratings.head()
book_tags.head()
tags.head()
books.shape
books.describe()
books.info()
!pip install Goodreads
from goodreads import client

api_key = 'k8aNpms0tdzaddORWzUHoA'

api_secret = '2Vy3eO4Nm2amvPLCxwRaufXwqKvd2pmS2E5FvNkXgg4'

gc = client.GoodreadsClient(api_key, api_secret)
null_year = books[books.original_publication_year.isna()==True].book_id

null_year.head()
for index,book_id in zip(null_year.index,null_year.values):

    books['original_publication_year'].iloc[index] = float(gc.book(book_id).publication_date[2])
# null_language = books[books.language_code.isna()==True].book_id



# for index,book_id in zip(null_language.index,null_language.values):

#     books['language_code'].iloc[index] = gc.book(book_id).language_code
# books['publisher'] = books.apply(lambda x: gc.book(x.book_id).publisher
books[books.title.duplicated(keep=False)==True].sort_values('title').transpose()
def plot_books_by(df, col, n_rows=10):

    plt.figure(figsize=(12,7))

    ax = sns.barplot(x=df[col].head(n_rows), y=df['title'].head(n_rows), data=df)

    plt.title('Best ' + str(n_rows) + ' books by ' + col.replace('_',' ').capitalize(), weight='bold')

    plt.xlabel('Score of ' + col)

    plt.ylabel('Book Title')    
important_columns = ['title','authors','average_rating','ratings_count','work_text_reviews_count']
book_sorted_ratings = books[books['ratings_count']>=1000].sort_values('average_rating', ascending=False)

book_sorted_ratings[important_columns].head(15)
plot_books_by(book_sorted_ratings, 'average_rating', 15)
book_sorted_ratings_count = books.sort_values('ratings_count', ascending=False)

book_sorted_ratings_count[important_columns].head(15)
plot_books_by(book_sorted_ratings_count, 'ratings_count', 15)
book_sorted_reviews_count = books[books['ratings_count']>=1000].sort_values('work_text_reviews_count', ascending=False)

book_sorted_reviews_count[important_columns].head(15)
plot_books_by(book_sorted_reviews_count, 'work_text_reviews_count', 15)
from sklearn.preprocessing import MinMaxScaler
cols = ['ratings_count','average_rating']

scaler = MinMaxScaler()

df_normalized = books[important_columns].copy()

df_normalized[cols] = scaler.fit_transform(df_normalized[cols])
df_normalized.head()
df_normalized['book_score'] = 0.5 * df_normalized['ratings_count'] + 0.5 * df_normalized['average_rating']
df_normalized_sorted_score = df_normalized.sort_values('book_score', ascending=False)

df_normalized_sorted_score.head()
plot_books_by(df_normalized_sorted_score, 'book_score', 15)
lang_counts = pd.DataFrame(books['language_code'].value_counts())

lang_counts.columns = ['counts']

lang_counts
len(lang_counts)
plt.figure(figsize=(16,8))

plt.title("Number of Books released in a specific Language (English included).", weight='bold')

plt.bar(x=lang_counts.index,height='counts', data=lang_counts);
lang_counts = lang_counts.drop(["en-US", "en-GB", "eng", "en-CA"])
plt.figure(figsize=(16,8))

plt.title("Number of Books released in a specific Language (English excluded).", weight='bold')

plt.bar(x=lang_counts.index,height='counts', data=lang_counts);
books['original_publication_year'] = books['original_publication_year'].astype(int)

year_count = books.groupby('original_publication_year')['title'].count()

plt.figure(figsize=(18,5))

year_count.plot();
plt.figure(figsize=(18,5))

year_count[year_count.index > 1900].plot();
books.groupby('original_publication_year')['title'].count()[books.groupby('original_publication_year')['title'].count()>200]
books[['title', 'original_publication_year']].sort_values('original_publication_year').head(10)
books[['book_id', 'title', 'original_publication_year']].sort_values('original_publication_year').tail(10)
books.authors.value_counts()
c = []

books.authors.apply(lambda x: c.append(x) if 'Agatha Christie' in x else [])

c
print('The real number of books (int the data) for Agatha Christie is {} not {}.'.format(len(c),books.authors.value_counts()['Agatha Christie']))
a = []

books.authors.apply(lambda x: a.append(x) if ',' in x else [])

a[:10]
print('Number of books with multi-authors is {}.'.format(len(a)))
authors_list = books['authors'].apply(lambda x: [a for a in x.split(', ')] if ',' in x else x)

authors_list.head()
authors_list.head(6).apply(lambda x: pd.Series(x)).stack().head(6)
authors_list.head(6).apply(lambda x: pd.Series(x)).stack().head(6).reset_index(level=1, drop=True)
splitted_authors = authors_list.apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True)

splitted_authors.name = 'authors'

splitted_authors.head()
df_edited_authors = books[important_columns].drop('authors', axis=1).join(splitted_authors)

df_edited_authors.head()
books.shape, df_edited_authors.shape
df_edited_authors.authors.value_counts()
def plot_authors_by(df, title, xlabel, n=15, ylabel='Author', y_size=7):

    plt.figure(figsize=(15,y_size))

    ax = sns.barplot(x=df.head(n).values, y=df.head(n).index)

    plt.title(title, weight='bold')

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)
authors_most_with_books = df_edited_authors.authors.value_counts()

authors_most_with_books.head(15)
plot_authors_by(authors_most_with_books, 'Authors with most books', 'Number of Books', 30)
def at_least_books(df, n):

    more_than_n = df_edited_authors['authors'].value_counts().values >= n

    return df['authors'].isin(df['authors'].value_counts()[more_than_n == True].index)
def at_least_ratings_count(df, n):

    return df.ratings_count >= n
at_least_books(df_edited_authors, 5).head()
at_least_ratings_count(df_edited_authors, 10000).head()
df_edited_authors['authors'].value_counts().describe()
plt.figure(figsize=(10,5))

plt.hist(df_edited_authors.authors.value_counts().values);

plt.xlabel('number of books');

plt.ylabel('count');
plt.figure(figsize=(17,2))

sns.boxplot(x=df_edited_authors.authors.value_counts(), orient='h');

plt.xlabel('number of books');
df_edited_authors['ratings_count'].describe()
plt.figure(figsize=(10,5))

plt.hist(df_edited_authors['ratings_count']);

plt.xlabel('number of ratings');

plt.ylabel('count');
plt.figure(figsize=(17,2))

sns.boxplot(x=df_edited_authors['ratings_count'], orient='h');
c1 = at_least_books(df_edited_authors, 5)

c2 = at_least_ratings_count(df_edited_authors, 50000)
best_rating_authors = df_edited_authors[c1 & c2].groupby('authors')['average_rating'].mean().sort_values(ascending=False)

best_rating_authors.head(15)
print('Best Rating Authors are {} of {} authors'.format(len(best_rating_authors),len(df_edited_authors.authors.unique())))
plot_authors_by(best_rating_authors, 'Best Rating Authors', 'Rating', 30)
most_ratings_authors = df_edited_authors.groupby('authors')['ratings_count'].sum().sort_values(ascending=False)

most_ratings_authors.head(15)
plot_authors_by(most_ratings_authors, 'Most Ratings Authors', 'Total Ratings', 30)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
best_rating_authors_normalized = pd.DataFrame(best_rating_authors.values, index=best_rating_authors.index, columns=['rating'])

best_rating_authors_normalized['rating'] = scaler.fit_transform(best_rating_authors_normalized)

best_rating_authors_normalized.head()
authors_rating_score = df_edited_authors['authors'].apply(lambda x: best_rating_authors_normalized.loc[x].rating if x in best_rating_authors_normalized.index else 0.0)

authors_rating_score.head()
most_ratings_authors_normalized = pd.DataFrame(most_ratings_authors.values, index=most_ratings_authors.index, columns=['total_rating'])

most_ratings_authors_normalized['total_rating'] = scaler.fit_transform(most_ratings_authors_normalized)

most_ratings_authors_normalized.head()
authors_ratings_count_score = df_edited_authors['authors'].apply(lambda x: most_ratings_authors_normalized.loc[x].total_rating)

authors_ratings_count_score.head()
df_edited_authors['author_score'] = 0.5 * authors_rating_score + 0.5 * authors_ratings_count_score

df_edited_authors['author_score'].head()
best_authors = df_edited_authors.groupby('authors')['author_score'].mean().sort_values(ascending=False)

best_authors.head(15)
plot_authors_by(best_authors, 'Best Authors', 'Score', 30)
ratings.head()
print('There are {} ratings.'.format(ratings.shape[0]))
print('Number of users is {}'.format(len(ratings.user_id.unique())))
ratings[ratings.duplicated()==True]
ratings[ratings.duplicated(keep=False)==True].head(6)
ratings_rmv_duplicates = ratings.drop_duplicates()

ratings_rmv_duplicates.shape
ratings_sample = ratings_rmv_duplicates.sample(frac=0.2)

ratings_sample.shape
plt.figure(figsize=(12,8))

sns.countplot(x='rating', data=ratings_sample);
ratings_per_user = ratings_sample.groupby('user_id')['user_id'].count()

ratings_per_user
plt.figure(figsize=(12,8))

plt.hist(ratings_per_user ,bins='auto');

plt.grid(axis='y', alpha=0.75)

plt.xlabel('number of ratings per user')

plt.ylabel('count');
mean_rating_per_user = ratings_sample.groupby('user_id')['rating'].mean()

mean_rating_per_user
plt.figure(figsize=(12,8))

plt.hist(mean_rating_per_user, bins='auto');

plt.grid(axis='y', alpha=0.75)

plt.xlabel('mean user rating')

plt.ylabel('count');
ratings_per_book = ratings_sample.groupby('book_id')['book_id'].count()

ratings_per_book
plt.figure(figsize=(12,8))

plt.hist(ratings_per_book ,bins='auto');

plt.grid(axis='y', alpha=0.75)

plt.xlabel('number of ratings per book')

plt.ylabel('count');
mean_rating_per_book = ratings_sample.groupby('book_id')['rating'].mean()

mean_rating_per_book
plt.figure(figsize=(12,8))

plt.hist(mean_rating_per_book, bins='auto');

plt.grid(axis='y', alpha=0.75)

plt.xlabel('mean book rating')

plt.ylabel('count');
tags.head()
tags.shape
book_tags.head()
book_tags.shape
genres = ["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics",

          "Comics", "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction",

          "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror",

          "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal",

          "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction", 

          "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]
genres = list(map(str.lower, genres))

genres[:4]
available_genres = tags.loc[tags.tag_name.str.lower().isin(genres)]
print('Number of available tags is {} out of the {} tags in genres list'.format(available_genres.shape[0], len(genres)))
available_books_with_genres = book_tags[book_tags.tag_id.isin(available_genres.tag_id)]

available_books_with_genres.shape
tag_counts = pd.DataFrame(available_books_with_genres.groupby('tag_id')['count'].sum())

tag_counts.head()
tag_counts.set_index(available_genres.tag_name, inplace=True)

tag_counts.head()
tag_counts.sort_values('count', ascending=False, inplace=True)

tag_counts.head()
plt.figure(figsize=(12,8))

sns.barplot(x='count', y=tag_counts.index, data=tag_counts, orient='h');
cols = ['books_count','original_publication_year','average_rating','ratings_count','work_ratings_count','work_text_reviews_count']
corr = books[cols].corr()

corr
plt.figure(figsize=(13,8))

sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm', annot=True);
mask = np.zeros(corr.shape, dtype=bool)

mask[np.triu_indices(len(mask))] = True



plt.figure(figsize=(13,8))

sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm', annot = True, mask = mask);
jp = sns.jointplot(x='ratings_count', y='average_rating', kind='reg', line_kws={'color':'cyan'}, data=books)

jp.annotate(stats.pearsonr, fontsize=12);
jp = sns.jointplot(x='books_count', y='average_rating', kind='reg', line_kws={'color':'cyan'}, data=books)

jp.annotate(stats.pearsonr, fontsize=12);
jp = sns.jointplot(x=ratings.groupby('user_id')['user_id'].count(),

                   y=ratings.groupby('user_id')['rating'].mean(),

                   kind='reg', line_kws={'color':'cyan'})

jp.annotate(stats.pearsonr, fontsize=12);

plt.xlabel('number_of_rated_books');

plt.ylabel('mean_rating');
subtitle = 'the lord of the rings'

books[books.title.str.lower().str.find(subtitle) > -1][['title']].values
book_series = books[books.title.str.contains('\(.*[,:]') == True][['title', 'average_rating']]
print('There are {} books that are in series'.format(book_series.shape[0]))
book_series.head()
series = book_series.copy()

series['title'] = series['title'].str.findall('\(.*[,:]').apply(lambda x: x.pop()[1:-1])
series.head()
jp = sns.jointplot(x=series.groupby('title')['title'].count(),

                   y=series.groupby('title')['average_rating'].mean(),

                   kind='reg', line_kws={'color':'cyan'})

jp.annotate(stats.pearsonr, fontsize=12);

plt.xlabel('number_of_volumes_in_series');

plt.ylabel('mean_rating');
title_length = books[['title', 'average_rating']]

title_length['length'] = title_length['title'].str.findall('\s').apply(lambda x: len(x)+1)
title_length.head()
plt.figure(figsize=(12,8))

sns.boxplot(x='length', y='average_rating', data=title_length);

plt.xlabel('title_length');

plt.grid()
has_subtitle = books[['title', 'average_rating']]

has_subtitle['has_subtitle'] = title_length['title'].str.contains(':', regex=False)
has_subtitle.head()
plt.figure(figsize=(6,8))

sns.boxplot(x='has_subtitle', y='average_rating', data=has_subtitle);

plt.grid()
n_of_authors = books.authors.apply(lambda x: len(x.split(',')))
plt.figure(figsize=(8,7))

jp = sns.regplot(x=n_of_authors, y=books['average_rating'], line_kws={'color':'cyan'})

jp.annotate('r =' + str(round(stats.pearsonr(n_of_authors,books['average_rating'])[0], 3)),

            xy=(20,3), fontsize=15);

plt.xlabel('number_of_authors');
