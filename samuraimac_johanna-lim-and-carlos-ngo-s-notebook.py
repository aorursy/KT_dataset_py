
import pandas as pd
import os
os.listdir('../input')
reviews = pd.read_csv('../input/reviews.csv')
reviews.head()
reviews.isnull().sum() 
reviews[['author_type']] = reviews[['author_type']].fillna("Unknown") 
reviews.isnull().sum() 
reviews[['title', 'artist']] = reviews[['title', 'artist']].fillna("Unknown")
reviews.isnull().sum() 
reviews = reviews[reviews['artist'].str.contains(',') == False]
reviews
reviews['score'].value_counts().sort_index().plot.line()
reviews['score'].mean()
reviews = reviews[reviews['artist'].str.contains(',') == True]
reviews
reviews['score'].value_counts().sort_index().plot.line()
reviews['score'].mean()