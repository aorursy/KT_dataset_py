import pandas as pd
import numpy as np
# Import CSV
book = pd.read_csv("../input/books.csv")
book_tags = pd.read_csv("../input/book_tags.csv")
ratings = pd.read_csv("../input/ratings.csv")
tags = pd.read_csv("../input/tags.csv")
to_read = pd.read_csv("../input/to_read.csv")
print(book.head(5))
print(tags.head(5))
print(ratings.head(5))
print(book_tags.head(5))
print(to_read.head(5))
#Ratings matrix to be one row per user and one column per movie.
print("Total Books: ", book.shape[0])
## Total User only considering user who rated atleast one movie
print("Total User: ", len(ratings['user_id'].unique()))
## Using Explicit feedback data only
books = book[['book_id', 'authors', 'original_title', 'average_rating']].copy()
books_df = ratings.reset_index().pivot_table(index = 'user_id', columns = 'book_id', values = 'rating').fillna(0)
print(books_df.head(), books_df.shape)
# Normalize by each users mean and convert it from a dataframe to a numpy array
R = books_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)
del(R_demeaned)
sigma = np.diag(sigma)
print(sigma)
# add the user means back to get the predicted 5-star ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
print(all_user_predicted_ratings)
# Making Recommendation
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = books_df.columns)
print(preds_df.head(5))
def recommend_books(predictions_df, userID, books_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.user_id == (userID)]
    user_full = (user_data.merge(books_df, how = 'left', left_on = 'book_id', right_on = 'book_id').
                     sort_values(['rating'], ascending=False)
                 )

#     print 'User {0} has already rated {1} books.', %(.format(userID, user_full.shape[0]))
#     print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating books that the user hasn't read yet.
    recommendations = (books_df[~books_df['book_id'].isin(user_full['book_id'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'book_id',
               right_on = 'book_id').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return(user_full, recommendations)
# Now we can recommend books to any user id (ex : user_id : 121)
already_rated, predictions = recommend_books(preds_df, 121, books, ratings, 10)
print(predictions, already_rated.shape)