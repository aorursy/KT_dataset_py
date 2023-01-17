import numpy as np #

import pandas as pd 

import os
INPUT_DIR = os.path.join("..", "input", "bible")

genre = pd.read_csv(os.path.join(INPUT_DIR, "key_genre_english.csv"), index_col='g', 

                    dtype={'g': np.int32, 'n': object})

book = pd.read_csv(os.path.join(INPUT_DIR, "key_english.csv"), index_col='b', 

                   dtype={'b': np.int32, 'n': object, 't': object, 'g': np.int32})

verse = pd.read_csv(os.path.join(INPUT_DIR, "t_asv.csv"), index_col='id', 

                    dtype={'id': np.int64, 'b': np.int32, 'c': np.int32, 'v': np.int32, 't': object})



verse.head()
bible = verse.join(book, on='b', lsuffix='_verse', rsuffix='_book').join(genre, on='g', lsuffix='_book', rsuffix='_genre')

bible = bible.rename(columns={

    'b': 'book_id', 

    'c': 'chapter', 

    'v': 'verse', 

    't_verse': 'text',

    'n_book': 'book', 

    't_book': 'testament', 

    'g': 'genre_id', 

    'n_genre': 'genre'})

bible.head()
bible.shape
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_df = 0.5, max_features = 1000)

X = vectorizer.fit_transform(bible.text)

X.shape
from sklearn.decomposition import TruncatedSVD



svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)

X = svd.fit_transform(X)

X.shape
cluster_data = pd.DataFrame({'comp1': X[:,0], 'comp2': X[:,1], 'book': bible.book, 'genre': bible.genre})

cluster_data.head()
from matplotlib import pyplot as plt

import seaborn as sns



sns.set(rc={'figure.figsize':(20, 20)})

sns.scatterplot('comp1', 'comp2', data=cluster_data, hue='book').set_title('By Book')

plt.show()
sns.set(rc={'figure.figsize':(20, 20)})

sns.scatterplot('comp1', 'comp2', data=cluster_data, hue='genre').set_title('By Genre')

plt.show()