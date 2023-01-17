import pickle

import pandas as pd

import numpy as np

import os, sys, gc 

from plotnine import *

import plotnine



from tqdm import tqdm_notebook

import seaborn as sns

import warnings

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

import matplotlib as mpl

from matplotlib import rc

import re

from matplotlib.ticker import PercentFormatter

import datetime

from math import log # IDF 계산을 위해
path = "../input/t-academy-recommendation/"
# pd.read_json : json 형태의 파일을 dataframe 형태로 불러오는 코드 

magazine = pd.read_json(path + 'magazine.json', lines=True) # lines = True : Read the file as a json object per line.

metadata = pd.read_json(path + 'metadata.json', lines=True)

users = pd.read_json(path + 'users.json', lines=True)
%%time 

import itertools

from itertools import chain

import glob

import os 





def chainer(s):

    return list(itertools.chain.from_iterable(s))
read_rowwise = pd.read_csv(path + "read_rowwise.csv")
from datetime import datetime 



metadata['reg_datetime'] = metadata['reg_ts'].apply(lambda x : datetime.fromtimestamp(x/1000.0))

metadata.loc[metadata['reg_datetime'] == metadata['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)

metadata['reg_dt'] = metadata['reg_datetime'].dt.date

metadata['type'] = metadata['magazine_id'].apply(lambda x : '개인' if x == 0.0 else '매거진')

metadata['reg_dt'] = pd.to_datetime(metadata['reg_dt'])
read_rowwise = read_rowwise.merge(metadata[['id', 'reg_dt']], how='left', left_on='article_id', right_on='id')

read_rowwise = read_rowwise[read_rowwise['article_id'] != '']



# 사용자가 읽은 글의 목록들을 저장 

read_total = pd.DataFrame(read_rowwise.groupby(['user_id'])['article_id'].unique()).reset_index()

read_total.columns = ['user_id', 'article_list']
read_rowwise = read_rowwise[(read_rowwise['id'].notnull()) & (read_rowwise['reg_dt'].notnull())]

read_rowwise = read_rowwise[(read_rowwise['reg_dt'] >= '2019-01-01') & (read_rowwise['reg_dt'] < '2090-12-31')].reset_index(drop=True)

del read_rowwise['id']
user2idx = {}

for i, l in enumerate(read_rowwise['user_id'].unique()):

    user2idx[l] = i

    

article2idx = {}

for i, l in enumerate(read_rowwise['article_id'].unique()):

    article2idx[l] = i
print(len(user2idx), len(article2idx))
valid = pd.read_csv(path + '/predict/dev.users', header=None)
read_total_valid = read_total[read_total['user_id'].isin(valid[0])].reset_index(drop=True)

read_total_train = read_total[~read_total['user_id'].isin(valid[0])].reset_index(drop=True)
read_total_train['article_len'] = read_total_train['article_list'].apply(lambda x: len(x))

top10_percent = np.percentile(read_total_train['article_len'].values, 90)

read_total_train = read_total_train[read_total_train['article_len'] >= top10_percent]

hot_user = read_total_train['user_id'].unique()



user_total = pd.DataFrame(read_rowwise.groupby(['article_id'])['user_id'].unique()).reset_index()

user_total.columns = ['article_id', 'user_list']



user_total['user_len'] = user_total['user_list'].apply(lambda x: len(x))

cold_article = user_total[user_total['user_len'] <= 20]['article_id'].unique()
read_rowwise = read_rowwise[read_rowwise['user_id'].isin(np.append(hot_user, valid[0].values))]

read_rowwise = read_rowwise[~read_rowwise['article_id'].isin(cold_article)]
user2idx = {}

for i, l in enumerate(read_rowwise['user_id'].unique()):

    user2idx[l] = i

    

article2idx = {}

for i, l in enumerate(read_rowwise['article_id'].unique()):

    article2idx[l] = i
idx2user = {i: user for user, i in user2idx.items()}

idx2article = {i: item for item, i in article2idx.items()}
print(len(user2idx), len(article2idx))
data = read_rowwise[['user_id', 'article_id']].reset_index(drop=True)

useridx = data['useridx'] = read_rowwise['user_id'].apply(lambda x: user2idx[x]).values

articleidx = data['articleidx'] = read_rowwise['article_id'].apply(lambda x: article2idx[x]).values

rating = np.ones(len(data))
import scipy



purchase_sparse = scipy.sparse.csr_matrix((rating, (useridx, articleidx)), shape=(len(set(useridx)), len(set(articleidx))))
from implicit.evaluation import *

from implicit.als import AlternatingLeastSquares as ALS
als_model = ALS(factors=20, regularization=0.08, iterations = 20)

als_model.fit(purchase_sparse.T)
als_model.recommend(0, purchase_sparse, N=150)[0:10]
popular_rec_model = read_rowwise['article_id'].value_counts().index[0:1000]



with open('recommend.txt', 'w') as f:

    for user in tqdm_notebook(valid[0].values):

        # 추천 후보 

        seen = chainer(read_total[read_total['user_id'] == user]['article_list'].values)

        

        try:

            recs = als_model.recommend(user2idx[user], purchase_sparse, N=150)

            recs = [idx2article[x[0]] for x in recs][0:100]          

            f.write('%s %s\n' % (user, ' '.join(recs)))

        except:

            recs = popular_rec_model[0:100]

            f.write('%s %s\n' % (user, ' '.join(recs)))