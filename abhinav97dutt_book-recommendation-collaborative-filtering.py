import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

books = pd.read_csv('../input/books.csv')
books

books['original_title'].isnull().sum()
books['book_id'].isnull().sum()
ratings = pd.read_csv('../input/ratings.csv')
ratings

ratings.apply(lambda x: x.isnull().sum(),axis=0)
books.apply(lambda x:x.isnull().sum(), axis=0)
books['title']
books_dataset = pd.DataFrame(books, columns=['book_id', 'authors', 'title', 'average_rating'])
books_dataset = books_dataset.sort_values('book_id')
books_dataset['book_id']
books_data = pd.merge(books_dataset, ratings, on='book_id')
books_data
each_book_rating = pd.pivot_table(books_data, index='user_id', values='rating', columns='title', fill_value=0)
each_book_rating
book_corr = np.corrcoef(each_book_rating.T)
book_corr.shape
book_list=  list(each_book_rating)

book_titles =[] 

for i in range(len(book_list)):

    book_titles.append(book_list[i])
book_titles
book = 'The Alchemist'

book_index = book_titles.index(book)
corr_score = book_corr[book_index]
print(sorted(corr_score, reverse=True))
condition = (corr_score >= 0.1)
np.extract(condition, book_titles)

# similar books to the alchemist
def get_recommendation(books_list):

    book_similarities = np.zeros(book_corr.shape[0])

    

    for book in books_list:    

#         print(book)

        book_index = book_titles.index(book)

#         print(book_index)

        book_similarities += book_corr[book_index] 

    book_preferences = []

    for i in range(len(book_titles)):

        book_preferences.append((book_titles[i],book_similarities[i]))

        

    return sorted(book_preferences, key= lambda x: x[1], reverse=True)

    

#     return book_preferences

my_fav_books = ['The Alchemist','The Adventures of Sherlock Holmes','The Great Gatsby','To Kill a Mockingbird','The Da Vinci Code (Robert Langdon, #2)','The Fellowship of the Ring (The Lord of the Rings, #1)']
book_recommendations = get_recommendation(my_fav_books)
print('The books you should like')

print('-'*25)

i=0

cnt=0

while cnt < 9:

    book_to_read = book_recommendations[i][0]

    i += 1

    if book_to_read in my_fav_books:

        continue

    else:

        print(book_to_read)

        cnt += 1

    
# book_recommendations