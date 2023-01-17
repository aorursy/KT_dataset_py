import gc

import os

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from multiprocessing import Pool



import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PATH="../input/"

os.listdir(PATH)
def merge_train_test(df_train, df_test):

    if "target" not in df_test.columns.values:

        df_test["target"] = -1

    res = pd.concat([df_train, df_test])

    res.reset_index(inplace=True, drop=True)

    return res



def split_train_test(df):

    df_train = df[df["target"] >= 0]

    df_test = df[df["target"] <= -1]

    df_train.reset_index(inplace=True, drop=True)

    df_test.reset_index(inplace=True, drop=True)

    assert list(df_train["ID_code"].values) == [f"train_{i}" for i in range(200000)]

    assert list(df_test["ID_code"].values) == [f"test_{i}" for i in range(200000)]

    return df_train, df_test
%%time

train_df = pd.read_csv(PATH+"train.csv")

test_df = pd.read_csv(PATH+"test.csv")
class CountEncoder:

    def fit(self, series):

        self.counts = series.groupby(series).count()

    

    def transform(self, series):

        return series.map(self.counts).fillna(0).astype(np.int16)
%%time

df_merged = merge_train_test(train_df, test_df)

for i in range(200):

    enc = CountEncoder()

    var = df_merged[f"var_{i}"]

    enc.fit(var)

    df_merged[f"{i}_count_enc"] = enc.transform(var)
train_df, test_df = split_train_test(df_merged)

for v in range(10):

    cnt_mean_trn = train_df[f"{v}_count_enc"].mean()

    cnt_mean_test = test_df[f"{v}_count_enc"].mean()

    print(f"cnt_mean_trn={cnt_mean_trn:2.5f}, cnt_mean_test={cnt_mean_test:2.5f}, diff={cnt_mean_trn-cnt_mean_test:.5f}")
%%time

df_merged["count_enc_1s"] = 0

for v in range(200):

    df_merged["count_enc_1s"] += (df_merged[f"{v}_count_enc"]==1)

train_df, test_df = split_train_test(df_merged)
plt.figure(figsize=(20, 4))

plt.hist(train_df["count_enc_1s"], range=(-0.5, 99.5), bins=100, alpha=0.5, label="train")

plt.hist(test_df["count_enc_1s"], range=(-0.5, 99.5), bins=100, alpha=0.5, label="test")

plt.legend()

plt.show()
(test_df["count_enc_1s"]==0).sum()
test_df["target"] = -1 - (test_df["count_enc_1s"] == 0)  # -1: public?  -2: private?

test_df.head()



# actually -1 is real and -2 is fake
%%time

for val, df_grouped in test_df.groupby(f"var_{v}"):

    if -2 in df_grouped["target"].values:

        assert -1 in df_grouped["target"].values

print("ok")
# unionfind tree

class Uf:

    def __init__(self, N):

        self.Par = list(range(N))



    def root(self, x):

        if self.Par[x] == x:

            return x

        else:

            self.Par[x] = self.root(self.Par[x])

            return self.Par[x]



    def same(self, x, y):

        return self.root(x) == self.root(y)



    def unite(self, x, y):

        x = self.root(x)

        y = self.root(y)

        if x != y:

            self.Par[x] = y
%%time

from itertools import groupby

from operator import itemgetter



index = range(200000)

target = test_df["target"].values



uf = Uf(200000)

for v in range(200):

    data = test_df[f"var_{v}"].values

    for k, g in groupby(sorted(zip(data, target, index)), key=itemgetter(0)):  # grouping by raw value

        g = list(g)

        if [tgt for _, tgt, _ in g].count(-1) == 1:  # if the value is unique in public? data

            idx0 = g[0][2]

            for _, _, idx in g[1:]:

                uf.unite(idx0, idx)  # belong to same group
from collections import defaultdict

cnt = defaultdict(int)

for i in test_df.index.values:

    cnt[uf.root(i)] += 1

cnt
cnt = defaultdict(int)

for i in test_df[target==-1].index.values:

    cnt[uf.root(i)] += 1

cnt
cnt = defaultdict(int)

for i in test_df[target==-2].index.values:

    cnt[uf.root(i)] += 1

cnt
test_df["public_private"] = [uf.root(idx)==166779 for idx in range(200000)]

test_df.head()