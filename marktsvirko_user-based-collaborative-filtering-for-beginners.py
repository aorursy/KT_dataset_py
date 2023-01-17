import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
books_df = pd.read_csv('../input/goodbooks-10k-updated/books.csv')
ratings_df = pd.read_csv('../input/goodbooks-10k-updated/ratings.csv')
books_df.info()
books_df.sample(5)
ratings_df.info()
ratings_df.sample(5)
my_list = {'Martin Eden': 5,
            'Pet Sematary': 5,
            'One Hundred Years of Solitude': 5,
            'Ham on Rye': 5,
            'The Grapes of Wrath': 4, 
            "Cat's Cradle": 5,
            'Crime and Punishment': 4,
            'The Trial': 4}
# Create dataframe for new user (me)
user_books = pd.DataFrame(columns=['title', 'rating'], data=my_list.items())

# Add book_id from books_df
new_user = pd.merge(user_books, books_df, on='title', how='inner')
new_user = new_user[['book_id', 'title', 'rating']].sort_values(by='book_id')
new_user
other_users = ratings_df[ratings_df['book_id'].isin(new_user['book_id'].values)]
other_users
other_users['user_id'].nunique()
# Sort users by count of most mutual books with me
users_mutual_books = other_users.groupby(['user_id'])
users_mutual_books = sorted(users_mutual_books, key=lambda x: len(x[1]), reverse=True)
users_mutual_books[0]
top_users = users_mutual_books[:100]
# Pearson correlation
from scipy.stats import pearsonr

pearson_corr = {}

for user_id, books in top_users:
    # Books should be sorted
    books = books.sort_values(by='book_id')
    book_list = books['book_id'].values

    new_user_ratings = new_user[new_user['book_id'].isin(book_list)]['rating'].values 
    user_ratings = books[books['book_id'].isin(book_list)]['rating'].values
   
    corr = pearsonr(new_user_ratings, user_ratings)
    pearson_corr[user_id] = corr[0]
# Get top50 users with the highest similarity indices
pearson_df = pd.DataFrame(columns=['user_id', 'similarity_index'], data=pearson_corr.items())
pearson_df = pearson_df.sort_values(by='similarity_index', ascending=False)[:50]
pearson_df
# Get all books for these users and add weighted book's ratings
users_rating = pearson_df.merge(ratings_df, on='user_id', how='inner')
users_rating['weighted_rating'] = users_rating['rating'] * users_rating['similarity_index']
users_rating
# Calculate sum of similarity index and weighted rating for each book
grouped_ratings = users_rating.groupby('book_id').sum()[['similarity_index', 'weighted_rating']]
recommend_books = pd.DataFrame()

# Add average recommendation score
recommend_books['avg_reccomend_score'] = grouped_ratings['weighted_rating']/grouped_ratings['similarity_index']
recommend_books['book_id'] = grouped_ratings.index
recommend_books = recommend_books.reset_index(drop=True)

# Left books with the highest score
recommend_books = recommend_books[(recommend_books['avg_reccomend_score'] == 5)]
recommend_books
# Let's see our recomendations
recommendation = books_df[books_df['book_id'].isin(recommend_books['book_id'])][['authors', 'title', 'book_id']].sample(10)
recommendation
