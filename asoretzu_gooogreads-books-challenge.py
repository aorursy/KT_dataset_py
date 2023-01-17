import numpy as np  # Linear Algebra
import pandas as pd  # Data Processing
import plotly.express as px # Data Visualization

import os  # Dirnames and Filenames
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.options.display.float_format = '{:.2f}'.format
# We load the dataset and print the first 10 rows:

df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)
df.head(5)
print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')
df.info()
print('Are duplicated values in the dataset?: {}.'.format(df.duplicated().any()))
print('Number of total books: {}.'.format(df['title'].count()))
print('Number of unique books: {}.'.format(df['title'].value_counts().count()))

print("\nPrinting a sample of books with more than one entry:")
df['title'].value_counts().head(10)
# Lets filter only the values we will need.

df = df[['title', 'authors', 'average_rating', 'ratings_count', 'text_reviews_count']]
df.head(5)
# Geting the top 10 books with the highest raiting count.

rating_count = df.sort_values('ratings_count',ascending=False).head(10)
rating_count.head(5)
# Lets now plot the top 10 for better understanding.

fig = px.scatter(rating_count, x="ratings_count", y="average_rating", hover_name="title", color="average_rating", size="ratings_count", size_max=60)
fig.show()
# Geting the top 10 books with the highest text reviews count and raiting count.

text_reviews = df.sort_values(['text_reviews_count','ratings_count',],ascending=False).head(10)
text_reviews.head(5)
# Lets now plot the top 10 again, for better understanding.

fig = px.scatter(text_reviews, x="text_reviews_count", y="average_rating", hover_name="title", color="average_rating", size="ratings_count", size_max=60)
fig.show()
