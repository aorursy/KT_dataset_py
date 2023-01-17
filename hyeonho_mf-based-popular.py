from collections import Counter

from datetime import timedelta, datetime

import glob

from itertools import chain

import json

import os

import re

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pandas.plotting import register_matplotlib_converters

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

from tqdm import tqdm_notebook
path = '../input/kakao-arena-data-preprocess/'

read = pd.read_csv(path+'read.csv')

users = pd.read_csv(path+'users.csv')

dev = pd.read_csv(path+'dev.csv')

test = pd.read_csv(path+'test.csv')
from itertools import chain

def chainer(s):

    return list(chain.from_iterable(s.str.split(' ')))

read_cnt_by_user = read['article_id'].str.split(' ').map(len)

read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),

                         'hr': np.repeat(read['hr'], read_cnt_by_user),

                         'user_id': np.repeat(read['user_id'], read_cnt_by_user),

                         'article_id': chainer(read['article_id'])})

read_raw = read_raw.reset_index(drop=True)

read_raw['article'] = read_raw['article_id'].apply(lambda x: str(x).split('_')[0])

del read
read_raw2 = read_raw.loc[read_raw['dt']>=20190222, ['user_id', 'article_id', 'article']].reset_index(drop=True)
from sklearn.model_selection import KFold



from sklearn.decomposition import NMF

import scipy
read_raw2 = read_raw2.groupby(['user_id', 'article_id']).count().reset_index()

read_raw2['article'] = read_raw2['article_id'].apply(lambda x: str(x).split('_')[0])



users = users[users['following_list'].apply(lambda x: True if len(x)>2 else False)].reset_index(drop=True)

users = users[users['id'].isin(read_raw2['user_id'].unique())].reset_index(drop=True)



test = pd.merge(test, users[['following_list', 'id']], how='left', on='id')
metadata = pd.read_json('../input/kakao-arena-2nd-competition/metadata.json', lines=True)
read_raw2 = read_raw2[read_raw2['article_id'].isin(metadata['id'])].reset_index(drop=True)
popular_list = read_raw2['article_id'].value_counts().index
users['following_list'] = users['following_list'].apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', '))
def recommend(train, user_id, raw_data):



    user2idx = {user: i for i, user in enumerate(train['user_id'].unique())}

    idx2user = {i: user for user, i in user2idx.items()}



    article2idx = {game: i for i, game in enumerate(train['article_id'].unique())}

    idx2article = {i: game for game, i in article2idx.items()}



    user_idx = train['user_id'].apply(lambda x: user2idx[x]).values

    article_idx = train['article_id2'] = train['article_id'].apply(lambda x: article2idx[x]).values

    value = np.ones(len(train))



    n_users = len(train['user_id'].unique())

    n_articles = len(train['article_id'].unique())



    # sparse matrix

    purchases_sparse = scipy.sparse.csr_matrix((value, (user_idx, article_idx)), shape=(n_users, n_articles))



    model = NMF(n_components=20, init='random', l1_ratio=0.1, alpha=0.1, shuffle=True)

    W = model.fit_transform(purchases_sparse)

    H = model.components_



    recommend_score = W[user2idx[user_id]].dot(H)

    

    view_history = raw_data.loc[raw_data['user_id'].isin([user_id]), 'article_id'].unique()



    

    recommend_df = pd.DataFrame(np.concatenate([train['article_id'].unique().reshape(-1, 1), recommend_score.reshape(-1, 1)], 1))



    top_100_list = recommend_df[~recommend_df[0].isin(view_history)].sort_values(1, ascending=False).iloc[:100, 0].values



    return top_100_list
test['recommend'] = ''



for idx in tqdm_notebook(range(len(test))):

    

    user_id = test.loc[idx, 'id']

    try:

        follow = users.loc[users['id'].isin([user_id]), 'following_list'].values[0]



        # based on popular

        similar_read_raw = read_raw2[read_raw2['article'].isin(follow)].reset_index(drop=True)



        top_100 = recommend(similar_read_raw, user_id, read_raw)

    except:

        

        view_history = read_raw.loc[read_raw['user_id'].isin([user_id]), 'article_id'].unique()

        

        top_100 = popular_list[~popular_list.isin(view_history)][:100]



    if len(top_100) != 100:

        view_history = read_raw.loc[read_raw['user_id'].isin([user_id]), 'article_id'].unique()

        top_add = popular_list[(~popular_list.isin(view_history)) & (~popular_list.isin(top_100))][:100-len(top_100)]

        top_100 = np.append(top_100, top_add)

        

    test.loc[idx, 'recommend'] = ' '.join(top_100)
test['submit'] = test['id'] + ' ' + test['recommend']

test['submit'].apply(lambda x: len(x.split(' '))).value_counts()
test['submit'].to_csv('recommend.csv', index=False)