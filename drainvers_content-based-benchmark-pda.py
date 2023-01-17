import os



def list_all_files_in(dirpath):

    for dirname, _, filenames in os.walk(dirpath):

        for filename in filenames:

            print(os.path.join(dirname, filename))



list_all_files_in('../input')
# Dataframes

import pandas as pd



# Linear algebra

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# List shifting

from collections import deque



# Similarities between vectors

from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from sklearn.feature_extraction.text import TfidfVectorizer



# Recommender library

import surprise as sp

from surprise.model_selection import cross_validate, train_test_split



# Sparse matrices

from scipy.sparse import coo_matrix



# LightFM

from lightfm import LightFM

from lightfm.evaluation import precision_at_k



# Stacking sparse matrices

from scipy.sparse import vstack



# Displaying stuff

from IPython.display import display



# Converting from strings back to Python data

from ast import literal_eval



# NLP libraries

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec, TaggedDocument



import warnings; warnings.simplefilter('ignore')
test_pda2019_df = pd.read_csv('../input/pda2019/test-PDA2019.csv')
train_pda2019_df = pd.read_csv('../input/pda2019/train-PDA2019.csv')

train_pda2019_df.drop('timeStamp', inplace=True, axis=1)

train_pda2019_df['rating'] = train_pda2019_df['rating'].astype(float)

display(train_pda2019_df.head())

print('Empty fields:')

print(train_pda2019_df.isna().sum())
def append_genre(row):

    return row['tag'] + row['genres'].split('|')
content_pda2019_df = pd.read_csv('../input/pda2019/content-PDA2019.csv')

content_pda2019_df['title'] = content_pda2019_df['title'].str.replace('(\(\d\d\d\d\))', '')

content_pda2019_df['title'] = content_pda2019_df['title'].apply(lambda x: x.strip())

content_pda2019_df['tag'] = content_pda2019_df['tag'].apply(literal_eval).apply(lambda x: str(list(set(x))))

display(content_pda2019_df.head())

print('Empty fields:')

print(content_pda2019_df.isna().sum())
content_stack = content_pda2019_df[content_pda2019_df['genres'] != '(no genres listed)'].set_index('itemID').genres.str.split('|', expand=True).stack()

genres_explode = pd.get_dummies(content_stack).groupby(level=0).sum().reset_index()

display(genres_explode.head())

content_pda2019_df['genres_vector'] = genres_explode.iloc[:,1:].values.tolist()

content_pda2019_df.head()
print('Empty fields:')

print(content_pda2019_df.isna().sum())
# content_pda2019_df['visual_fvec'] = content_pda2019_df[['visual_f1', 'visual_f2', 'visual_f3', 'visual_f4', 'visual_f5', 'visual_f6', 'visual_f7']].values.tolist()
movie_tag_corpus = content_pda2019_df['tag'].values

stop_words = stopwords.words('english')
def tokenize_and_clean(doc):

    tokens = word_tokenize(doc.lower())

    tokens = [word for word in tokens if word.isalpha() and not word in stop_words]

    return tokens
movie_tag_doc = [TaggedDocument(words=tokenize_and_clean(doc), tags=[str(i)]) for i, doc in enumerate(movie_tag_corpus)]
params = {

    'max_epochs': 50,

    'vec_size': 9296,

    'alpha': 0.025,

    'min_count': 1,

    'dm': 0

}



model = Doc2Vec(**params)

model.build_vocab(movie_tag_doc)
print('Epoch', end=': ')

for epoch in range(params['max_epochs']):

    print(epoch, end=' ')

    model.train(movie_tag_doc, total_examples=model.corpus_count, epochs=model.epochs)

    model.alpha -= 0.0002

    model.min_alpha = model.alpha
movie_tag_vectors = model.docvecs.vectors_docs

movie_tag_vectors.shape
user_movies_df = pd.merge(train_pda2019_df, content_pda2019_df[['itemID', 'title']], on='itemID')

user_movies_df.head()
def recommend(row):

    user_movies = user_movies_df[user_movies_df['userID'] == row['userID']]['title'].tolist()

    user_movie_vector = np.zeros(shape=movie_tag_vectors.shape[1])



    for movie in user_movies:

        movie_indices = content_pda2019_df[content_pda2019_df['title'] == movie].index.values[0]

        user_movie_vector += movie_tag_vectors[movie_indices]



    user_movie_vector /= len(user_movies)

    # print(user_movie_vector)



    item_id_list = []

    similar_movies = model.docvecs.most_similar(positive=[user_movie_vector], topn=30)



    for i, j in similar_movies:

        movie_match = content_pda2019_df.loc[int(i), 'title'].strip()

        if movie_match not in user_movies:

            item_id_list.append(content_pda2019_df.loc[int(i), 'itemID'])

    

    return ' '.join([str(item_id) for item_id in item_id_list][:10])
submission_df = test_pda2019_df.copy(deep=True)

submission_df['recommended_itemIDs'] = submission_df.apply(recommend, axis=1)

submission_df.head()
submission_df.to_csv('submission-drainvers-PDA2019.csv', index=False)