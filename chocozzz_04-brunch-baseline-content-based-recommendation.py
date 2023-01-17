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
from sklearn.feature_extraction.text import TfidfVectorizer



metadata = metadata[metadata['keyword_list'].notnull()].reset_index()

metadata = metadata[metadata['reg_dt'] >= '2019-01-01']
user_total = pd.DataFrame(read_rowwise.groupby(['article_id'])['user_id'].unique()).reset_index()

user_total.columns = ['article_id', 'user_list']



user_total['user_len'] = user_total['user_list'].apply(lambda x: len(x))

cold_article = user_total[user_total['user_len'] <= 20]['article_id'].unique()

metadata = metadata[~metadata['id'].isin(cold_article)]
article2idx = {}

for i, l in enumerate(metadata['id'].unique()):

    article2idx[l] = i

    

idx2article = {i: item for item, i in article2idx.items()}

articleidx = metadata['articleidx'] = metadata['id'].apply(lambda x: article2idx[x]).values
import scipy



docs = metadata['keyword_list'].apply(lambda x: ' '.join(x)).values

tfidv = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None).fit(docs)

tfidv_df = scipy.sparse.csr_matrix(tfidv.transform(docs))

tfidv_df = tfidv_df.astype(np.float32)
print(tfidv_df.shape)
popular_rec_model = read_rowwise['article_id'].value_counts().index[0:100]
del read_rowwise

gc.collect()
from sklearn.metrics.pairwise import cosine_similarity



# 메모리 문제 발생 

cos_sim = cosine_similarity(tfidv_df, tfidv_df)
valid = pd.read_csv(path + '/predict/dev.users', header=None)
%%time 

top_n = 100

with open('./recommend.txt', 'w') as f:

    for user in tqdm_notebook(valid[0].values):

        seen = chainer(read_total[read_total['user_id'] == user]['article_list'])

        for seen_id in seen:

            # 2019년도 이전에 읽어서 혹은 메타데이터에 글이 없어서 유사도 계산이 안된 글

            cos_sim_sum = np.zeros(len(cos_sim))

            try:

                cos_sim_sum += cos_sim[article2idx[seen_id]]

            except:

                pass



        recs = []

        for rec in cos_sim_sum.argsort()[-(top_n+100):][::-1]:

            if (idx2article[rec] not in seen) & (len(recs) < 100):

                recs.append(idx2article[rec])



        f.write('%s %s\n' % (user, ' '.join(recs[0:100])))