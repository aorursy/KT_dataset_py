import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
books = pd.read_csv("/kaggle/input/goodreadsbooks/books.csv", error_bad_lines=False)

books.head().T
short_books = books[books['# num_pages'] <= 100]

short_books = short_books[short_books['average_rating'] > 3.0]

short_books.describe(include = 'all')
short_books = short_books.query('ratings_count > 50 and language_code=="eng"')

short_books.shape
short_books.sort_values('average_rating', ascending=False)
short_books[short_books['# num_pages'] > 0].sort_values('# num_pages')