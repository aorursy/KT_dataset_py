import pandas as pd
import numpy as np
import os
import gensim
from gensim.models import Word2Vec
import pickle
from joblib import load, dump
import torch

train_path = '../input/algotx/algo-tx/train_preliminary/'
test_path = '../input/algotx/algo-tx/test/'
train_ad = pd.read_csv(train_path + 'ad.csv')
test_ad = pd.read_csv(test_path + 'ad.csv')
merged_ad = pd.merge(train_ad, test_ad, on=['ad_id', 'creative_id', 'advertiser_id'], how='outer')
merged_ad = merged_ad.drop(['product_id_x', 'product_category_x', 'advertiser_id', 'industry_x', 'product_id_y', 'product_category_y', 'industry_y'], 1)
file1 = train_path + 'click_log.csv'
file2 = test_path + 'click_log.csv'
df_log = pd.concat([pd.read_csv(file1), pd.read_csv(file2)], ignore_index=True, axis=0).drop(
    ['time', 'click_times'], 1
)
df_log = df_log.sort_values(by=['user_id']).reindex()
df_log = pd.merge(df_log, merged_ad, on='creative_id').set_index('user_id').sort_index()
df_log
m = 1900000
feat = 'ad_id'
sentences = df_log.groupby(['user_id']).apply(lambda x: x[feat].tolist()).tolist()
sentences = [[str(j) for j in i] for i in sentences]
model = Word2Vec(sentences=sentences, min_count=1, sg=1, window=64, size=64, workers=12, iter=15)
print('w2v completed')
w2i = dict()
vocab_size = 0
for i in sentences:
    for j in i:
        if not j in w2i:
            w2i[j] = vocab_size
            vocab_size += 1
            if vocab_size % 1000000 == 0:
                print('processing', vocab_size)
X = torch.zeros((m, seq_len), dtype=torch.int)
xl = torch.zeros(m, dtype=torch.int)
for i, s in enumerate(sentences):
    if len(s) > seq_len:
        s = s[:seq_len]
    xl[i] = len(s)
    for j in range(xl[i]):
        X[i][j] = w2i[s[j]]
    if i % 500000 == 0:
        print('processing2', i)
emb = torch.zeros((m, 64), dtype=torch.float)
for k, v in w2i.items():
    emb[v] = model.wv[k]
dump((X, xl, emb), '{}.dat'.format(feat))