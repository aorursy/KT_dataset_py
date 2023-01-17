import os

import numpy as np

import pandas as pd



# Sklearn Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline



import matplotlib.pyplot as plt

import lightgbm as lgb
submission_df = pd.read_csv("../input/shopee-sentiment-analysis/sampleSubmission.csv")

test_df = pd.read_csv("../input/shopee-sentiment-analysis/test.csv")

train_df = pd.read_csv("../input/shopee-sentiment-analysis/train.csv")

submission_df.shape, test_df.shape, train_df.shape
# Optional step, still same output 4=5



# # Duplicated review but with different rating, if the mean is not integers

# # then that review has different rating

# diff_rating = train_df.groupby('review')['rating'].agg(['mean', 'count'])

# to_removed = diff_rating[diff_rating['mean'] % 1 != 0.0].index.to_list()



# # Since it is very small, we might just remove them

# print(f"Before removed shape: {train_df.shape}")

# print(f"To be removed: {diff_rating['count'].sum()}")

# train_df = train_df[~train_df['review'].isin(to_removed)].reset_index(drop=True)

# print(f"After removed shape: {train_df.shape}")
# Top n-gram correlation

def get_term_frequency(corpus, ngram_range=(1, 1)):

    tokenizer_kwargs = dict(

        analyzer='word',  # for many misspelled words, 'char_wb' is recommended

        ngram_range=ngram_range,  # (1, 1) = unigram, unigram, (1, 2) = unigram, bigram, etc.

        min_df=2,  # if integer, remove word that occurs less than this number

    )

    token_f = CountVectorizer(

        input='content',

        **tokenizer_kwargs,

    )    

    A_tokenized = token_f.fit_transform(corpus)

    

    term_count = np.array(A_tokenized.sum(axis=0)).flatten().tolist()

    term_names = token_f.get_feature_names()

    term_df = pd.DataFrame(list(zip(term_names, term_count)), columns=['name', 'count']).sort_values(by='count', ascending=False)

    term_df = term_df.set_index('name')

    

    return term_df



def plot_side_by_side(first_df, second_df, n_show=50, top=True):

    n_show = n_show

    fig, ax = plt.subplots(1, 2, figsize=(14, n_show/5))

    if top:

        first_df.head(n_show)[::-1].plot(kind='barh', ax=ax[0], legend=None, alpha=0.7)

        second_df.head(n_show)[::-1].plot(kind='barh', ax=ax[1], legend=None, alpha=0.7)

    else:

        first_df.tail(n_show).plot(kind='barh', ax=ax[0], legend=None, alpha=0.7)

        second_df.tail(n_show).plot(kind='barh', ax=ax[1], legend=None, alpha=0.7)

    ax[0].set_title(f'Rating=4 top {n_show} Terms')

    ax[1].set_title(f'Rating=5 top {n_show} Terms')

    ax[0].set_ylabel('')

    ax[1].set_ylabel('')

    plt.tight_layout()

    plt.show()
train_term_df_4 = get_term_frequency(train_df[train_df['rating'] == 4]['review'], ngram_range=(1, 1))

train_term_df_5 = get_term_frequency(train_df[train_df['rating'] == 5]['review'], ngram_range=(1, 1))

plot_side_by_side(train_term_df_4, train_term_df_5)
train_term_df_4 = get_term_frequency(train_df[train_df['rating'] == 4]['review'], ngram_range=(2, 2))

train_term_df_5 = get_term_frequency(train_df[train_df['rating'] == 5]['review'], ngram_range=(2, 2))

plot_side_by_side(train_term_df_4, train_term_df_5)
train_term_df_4 = get_term_frequency(train_df[train_df['rating'] == 4]['review'], ngram_range=(3, 3))

train_term_df_5 = get_term_frequency(train_df[train_df['rating'] == 5]['review'], ngram_range=(3, 3))

plot_side_by_side(train_term_df_4, train_term_df_5)
train_rating_check = train_df[train_df['rating'].isin([4, 5])].reset_index(drop=True)

train_rating_check['rating'] = train_rating_check['rating'] - 4  # make it 0, 1 for binary classification

base_mask = np.random.rand(len(train_rating_check)) < 0.6

train_mask = base_mask

valid_mask = ~base_mask

col_target = 'rating'
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(train_rating_check['review'])

X.shape
train_data = lgb.Dataset(X[train_mask],

                         label=train_rating_check[train_mask][col_target],)

valid_data = lgb.Dataset(X[valid_mask],

                         label=train_rating_check[valid_mask][col_target],)

params =  {

    "learning_rate": 0.1,

    "num_leaves": 100,

    "colsample_bytree": 0.75,

    "subsample": 0.75,

    "subsample_freq": 1,

    "max_depth": 2,

    "nthreads": 8,

    "verbose": 1,

    'metric': 'auc',

    'objective': 'binary',

    "early_stopping_rounds": 100,

    "reg_lambda": 0.5,

    "num_boost_round": 100000,

    "seed": 1234,

}



bst = lgb.train(params,

                train_data,

                valid_sets=[train_data, valid_data])
# Redo using our randomly assigned label data

train_rating_check['rating'] = np.random.randint(2, size=train_rating_check.shape[0])

train_rating_check['rating'].values
train_data = lgb.Dataset(X[train_mask],

                         label=train_rating_check[train_mask][col_target],)

valid_data = lgb.Dataset(X[valid_mask],

                         label=train_rating_check[valid_mask][col_target],)

params =  {

    "learning_rate": 0.1,

    "num_leaves": 100,

    "colsample_bytree": 0.75,

    "subsample": 0.75,

    "subsample_freq": 1,

    "max_depth": 2,

    "nthreads": 8,

    "verbose": 1,

    'metric': 'auc',

    'objective': 'binary',

    "early_stopping_rounds": 100,

    "reg_lambda": 0.5,

    "num_boost_round": 100000,

    "seed": 1234,

}



bst = lgb.train(params,

                train_data,

                valid_sets=[train_data, valid_data])