import numpy as np
import pandas as pd


books = pd.read_csv('../input/books.csv')
books.info()

book = books[['book_id','authors','title']]
book.head()
book.info()
ratings = pd.read_csv('../input/ratings.csv')
ratings.info()

ratings['rating'].unique()
books_data = pd.merge(book, ratings, on='book_id')
from surprise import Reader, Dataset, SVD, evaluate,accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

reader = Reader(rating_scale=(1,5))


data = Dataset.load_from_df(ratings[['book_id', 'user_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

#kf = KFold(n_splits=3)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions, verbose=True)
#for trainset, testset in kf.split(data):

    # train and test algorithm.
    #algo.fit(trainset)
    #predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    #accuracy.rmse(predictions, verbose=True)


def recommendation(user_id):
    user = book.copy()
    already_read = books_data[books_data['user_id'] == user_id]['book_id'].unique()
    user = user.reset_index()
    user = user[~user['book_id'].isin(already_read)]
    user['Estimate_Score']=user['book_id'].apply(lambda x: algo.predict(user_id, x).est)
    user = user.drop('book_id', axis = 1)
    user = user.sort_values('Estimate_Score', ascending=False)
    print(user.head(10))
recommendation(2)