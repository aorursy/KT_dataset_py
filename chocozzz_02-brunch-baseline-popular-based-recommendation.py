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
# 2019년도 이후로 작성된 글중에서 상위 100건의 글을 추천 

# 단, 이미 읽은 글의 경우는 추천에서 제외 

read_rowwise = read_rowwise.merge(metadata[['id', 'reg_dt']], how='left', left_on='article_id', right_on='id')
read_rowwise.head()
# 사용자가 읽은 글의 목록들을 저장 

read_total = pd.DataFrame(read_rowwise.groupby(['user_id'])['article_id'].unique()).reset_index()

read_total.columns = ['user_id', 'article_list']
# 1. article_id가 결측치인 경우는 삭제 (작가가 사라진 경우)

# 2. reg_dt가 결측치인 경우는 삭제 (메타데이터에 자료가 없는 경우)

read_rowwise = read_rowwise[read_rowwise['article_id'] != '']

read_rowwise = read_rowwise[(read_rowwise['id'].notnull()) & (read_rowwise['reg_dt'].notnull())]

read_rowwise = read_rowwise[(read_rowwise['reg_dt'] >= '2019-01-01') & (read_rowwise['reg_dt'] < '2090-12-31')].reset_index(drop=True)



del read_rowwise['id']
valid = pd.read_csv(path + '/predict/dev.users', header=None)
%%time 



popular_rec_model = read_rowwise['article_id'].value_counts().index[0:1000]

with open('recommend.txt', 'w') as f:

    for user in tqdm_notebook(valid[0].values):

        # 추천 후보 

        seen = chainer(read_total[read_total['user_id'] == user]['article_list'].values)

        recs = []

        for r in popular_rec_model:

            if len(recs) == 100: 

                break 

            else: 

                if r not in seen: recs.append(r)

        f.write('%s %s\n' % (user, ' '.join(recs)))
following_cnt_by_user = users['following_list'].map(len)

following_rowwise = pd.DataFrame({'user_id': np.repeat(users['id'], following_cnt_by_user),

                             'author_id': chainer(users['following_list'])})



following_rowwise.reset_index(drop=True, inplace=True)
following_rowwise = following_rowwise[following_rowwise['user_id'].isin(valid[0].values)]

following_rowwise.head()
%%time 

metadata_ = metadata[['user_id', 'id', 'reg_dt']]

metadata_.columns = ['author_id', 'article_id', 'reg_dt']

following_popular_model = pd.merge(following_rowwise, metadata_, how='left', on='author_id')
%%time 

read_rowwise['author_id'] = read_rowwise['article_id'].apply(lambda x: x.split('_')[0])

author_favor = read_rowwise.groupby(['user_id', 'author_id'])['author_id'].agg({'count'}).reset_index()
popular_model = pd.DataFrame(read_rowwise['article_id'].value_counts()).reset_index()

popular_model.columns = ['article_id', 'count']
following_popular_model = pd.merge(following_popular_model, author_favor, how='left', on=['user_id', 'author_id'])

following_popular_model = following_popular_model[following_popular_model['count'].notnull()].reset_index(drop=True)

following_popular_model = pd.merge(following_popular_model, popular_model, how='left', on='article_id')

following_popular_model.head()
following_popular_model = following_popular_model.sort_values(by=['count_x', 'count_y', 'reg_dt'], ascending=[False, False, False])
following_popular_model[following_popular_model['user_id'] == '#a6f7a5ff90a19ec4d583f0db1836844d'].head()
%%time 



with open('./recommend.txt', 'w') as f:

    for user in tqdm_notebook(valid[0].values):

        # 추천 후보 

        seen = chainer(read_total[read_total['user_id'] == user]['article_list'].values)

        following_rec_model = following_popular_model[following_popular_model['user_id'] == user]['article_id'].values

        recs = []

        for r in following_rec_model:

            if len(recs) == 100:

                break 

            else: 

                if r not in seen + recs: recs.append(r)

        

        if len(recs) < 100: 

            for r in popular_rec_model:

                if len(recs) == 100: 

                    break 

                else: 

                    if r not in seen + recs: recs.append(r)            

        f.write('%s %s\n' % (user, ' '.join(recs)))