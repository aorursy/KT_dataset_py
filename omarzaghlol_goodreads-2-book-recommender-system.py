import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime



import warnings

warnings.filterwarnings('ignore')
books = pd.read_csv('../input/goodbooks-10k//books.csv')

ratings = pd.read_csv('../input/goodbooks-10k//ratings.csv')

book_tags = pd.read_csv('../input/goodbooks-10k//book_tags.csv')

tags = pd.read_csv('../input/goodbooks-10k//tags.csv')
books['original_publication_year'] = books['original_publication_year'].fillna(-1).apply(lambda x: int(x) if x != -1 else -1)
ratings_rmv_duplicates = ratings.drop_duplicates()

unwanted_users = ratings_rmv_duplicates.groupby('user_id')['user_id'].count()

unwanted_users = unwanted_users[unwanted_users < 3]

unwanted_ratings = ratings_rmv_duplicates[ratings_rmv_duplicates.user_id.isin(unwanted_users.index)]

new_ratings = ratings_rmv_duplicates.drop(unwanted_ratings.index)
new_ratings['title'] = books.set_index('id').title.loc[new_ratings.book_id].values
new_ratings.head(10)
v = books['ratings_count']

m = books['ratings_count'].quantile(0.95)

R = books['average_rating']

C = books['average_rating'].mean()

W = (R*v + C*m) / (v + m)
books['weighted_rating'] = W
qualified  = books.sort_values('weighted_rating', ascending=False).head(250)
qualified[['title', 'authors', 'average_rating', 'weighted_rating']].head(15)
book_tags.head()
tags.head()
genres = ["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics",

          "Comics", "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction",

          "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror",

          "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal",

          "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction", 

          "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]
genres = list(map(str.lower, genres))

genres[:4]
available_genres = tags.loc[tags.tag_name.str.lower().isin(genres)]
available_genres.head()
available_genres_books = book_tags[book_tags.tag_id.isin(available_genres.tag_id)]
print('There are {} books that are tagged with above genres'.format(available_genres_books.shape[0]))
available_genres_books.head()
available_genres_books['genre'] = available_genres.tag_name.loc[available_genres_books.tag_id].values

available_genres_books.head()
def build_chart(genre, percentile=0.85):

    df = available_genres_books[available_genres_books['genre'] == genre.lower()]

    qualified = books.set_index('book_id').loc[df.goodreads_book_id]



    v = qualified['ratings_count']

    m = qualified['ratings_count'].quantile(percentile)

    R = qualified['average_rating']

    C = qualified['average_rating'].mean()

    qualified['weighted_rating'] = (R*v + C*m) / (v + m)



    qualified.sort_values('weighted_rating', ascending=False, inplace=True)

    return qualified
cols = ['title','authors','original_publication_year','average_rating','ratings_count','work_text_reviews_count','weighted_rating']
genre = 'Fiction'

build_chart(genre)[cols].head(15)
list(enumerate(available_genres.tag_name))
idx = 24  # romance

build_chart(list(available_genres.tag_name)[idx])[cols].head(15)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
books['authors'] = books['authors'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x.split(', ')])
def get_genres(x):

    t = book_tags[book_tags.goodreads_book_id==x]

    return [i.lower().replace(" ", "") for i in tags.tag_name.loc[t.tag_id].values]
books['genres'] = books.book_id.apply(get_genres)
books['soup'] = books.apply(lambda x: ' '.join([x['title']] + x['authors'] + x['genres']), axis=1)
books.soup.head()
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

count_matrix = count.fit_transform(books['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(books.index, index=books['title'])

titles = books['title']
def get_recommendations(title, n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    book_indices = [i[0] for i in sim_scores]

    return list(titles.iloc[book_indices].values)[:n]
get_recommendations("The One Minute Manager")
def get_name_from_partial(title):

    return list(books.title[books.title.str.lower().str.contains(title) == True].values)
title = "business"

l = get_name_from_partial(title)

list(enumerate(l))
get_recommendations(l[1])
def improved_recommendations(title, n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    book_indices = [i[0] for i in sim_scores]

    df = books.iloc[book_indices][['title', 'ratings_count', 'average_rating', 'weighted_rating']]



    v = df['ratings_count']

    m = df['ratings_count'].quantile(0.60)

    R = df['average_rating']

    C = df['average_rating'].mean()

    df['weighted_rating'] = (R*v + C*m) / (v + m)

    

    qualified = df[df['ratings_count'] >= m]

    qualified = qualified.sort_values('weighted_rating', ascending=False)

    return qualified.head(n)
improved_recommendations("The One Minute Manager")
improved_recommendations(l[1])
# ! pip install surprise
from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate
reader = Reader()

data = Dataset.load_from_df(new_ratings[['user_id', 'book_id', 'rating']], reader)
svd = SVD()

cross_validate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()

svd.fit(trainset);
new_ratings[new_ratings['user_id'] == 10]
svd.predict(10, 1506)
# bookmat = new_ratings.groupby(['user_id', 'title'])['rating'].mean().unstack()

bookmat = new_ratings.pivot_table(index='user_id', columns='title', values='rating')

bookmat.head()
def get_similar(title, mat):

    title_user_ratings = mat[title]

    similar_to_title = mat.corrwith(title_user_ratings)

    corr_title = pd.DataFrame(similar_to_title, columns=['correlation'])

    corr_title.dropna(inplace=True)

    corr_title.sort_values('correlation', ascending=False, inplace=True)

    return corr_title
title = "Twilight (Twilight, #1)"

smlr = get_similar(title, bookmat)
smlr.head(10)
smlr = smlr.join(books.set_index('title')['ratings_count'])

smlr.head()
smlr[smlr.ratings_count > 5e5].sort_values('correlation', ascending=False).head(10)
def hybrid(user_id, title, n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:51]

    book_indices = [i[0] for i in sim_scores]

    

    df = books.iloc[book_indices][['book_id', 'title', 'original_publication_year', 'ratings_count', 'average_rating']]

    df['est'] = df['book_id'].apply(lambda x: svd.predict(user_id, x).est)

    df = df.sort_values('est', ascending=False)

    return df.head(n)
hybrid(4, 'Eat, Pray, Love')
hybrid(10, 'Eat, Pray, Love')
def improved_hybrid(user_id, title, n=10):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:51]

    book_indices = [i[0] for i in sim_scores]

    

    df = books.iloc[book_indices][['book_id', 'title', 'ratings_count', 'average_rating', 'original_publication_year']]

    v = df['ratings_count']

    m = df['ratings_count'].quantile(0.60)

    R = df['average_rating']

    C = df['average_rating'].mean()

    df['weighted_rating'] = (R*v + C*m) / (v + m)

    

    df['est'] = df['book_id'].apply(lambda x: svd.predict(user_id, x).est)

    

    df['score'] = (df['est'] + df['weighted_rating']) / 2

    df = df.sort_values('score', ascending=False)

    return df[['book_id', 'title', 'original_publication_year', 'ratings_count', 'average_rating', 'score']].head(n)
improved_hybrid(4, 'Eat, Pray, Love')
improved_hybrid(10, 'Eat, Pray, Love')