import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



def show_percentage(number):

    percentage = "{:.3%}".format(number)

    return percentage



filepath = '/kaggle/input/corpus-de-libros-en-espanol/BooksInSpanishTokenCounts.0.01.csv'

books = pd.read_csv(filepath)



#This is the criteria that will be used for the data show. Examples of values include 'Source langage', 'Category', 'Author'

criteria = 'Category'

books['adverb%'] = books['# of adverbs']/books['Total tokens']



for value in books[criteria].unique():

    books_subset = books[books[criteria]==value]

    print('{} ({} books): max. {}, min. {}, max. {}, range {}'.format(value, len(books_subset), show_percentage(np.max(books_subset['adverb%'])), show_percentage(np.min(books_subset['adverb%'])), show_percentage(np.average(books_subset['adverb%'])), show_percentage(np.max(books_subset['adverb%']) - np.min(books_subset['adverb%']))))
