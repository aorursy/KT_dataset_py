import gc

import os

import random

import csv

import sys

import json



import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)



from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn import metrics

from tqdm import tqdm



plt.style.use("seaborn")

sns.set(font_scale=1)



sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
def prepare(df,test=False):

    keep_id = df[df.type == "Assessment"][['installation_id']].drop_duplicates()

    df = pd.merge(df, keep_id, on="installation_id", how="inner")

    attempt = df[((df.type=="Assessment") & (((df.event_code==4110) & (df.title=="Bird Measurer (Assessment)")) | ((df.event_code==4100) & (df.title!="Bird Measurer (Assessment)"))))]

    attempt.drop(["event_count", "event_code", "game_time"], axis=1, inplace=True)

    X = pd.DataFrame()

    for iid in tqdm(keep_id.installation_id):

        sample_id = attempt[attempt.installation_id == iid].reset_index()

        if(len(sample_id)==0):

            continue

        edata = pd.io.json.json_normalize(sample_id.event_data.apply(json.loads))

        sample_id = sample_id.join(edata)

        X = X.append(sample_id,ignore_index=True)

    X.drop(["event_data"],axis=1,inplace=True)

    X.drop(["index","buckets","buckets_placed","caterpillars","duration","left","pillars","right","stumps","timestamp"],axis=1,inplace=True)

    

    le = LabelEncoder()

    X.title = le.fit_transform(X.title)

    X.type = le.fit_transform(X.type)

    X.world = le.fit_transform(X.world)

    X.loc[X.correct==True,'target'] = 1

    X.loc[X.correct==False,'target'] = 0

    return X
X_train = prepare(train)

X_test = prepare(test,True)

print(X_train.columns)

print(X_test.columns)
X_test.reset_index().to_csv("results_in_test.csv",index=False)

X_test.reset_index().to_csv("results_in_train.csv",index=False)
def agg(df):

    df_agg = df.groupby(["installation_id","game_session"])["target"].agg({"num_collect":np.sum, "num_attempt":len})

    df_agg["accuracy"] = df_agg.num_collect/df_agg.num_attempt

    df_agg["accuracy_group"] = 0

    df_agg.loc[(df_agg.num_collect>0) & (df_agg.num_attempt==1),"accuracy_group"] = 3

    df_agg.loc[(df_agg.num_collect>0) & (df_agg.num_attempt==2),"accuracy_group"] = 2

    df_agg.loc[(df_agg.num_collect>0) & (df_agg.num_attempt>2),"accuracy_group"] = 1

    return df_agg
X_test_agg = agg(X_test)

X_train_agg = agg(X_train)
print(X_test_agg)
# X_test_sub = X_test_agg.groupby("installation_id").max()

X_test_sub = X_test_agg.copy()

X_test_sub.reset_index().to_csv("results_in_test_agg.csv",index=False)

# X_train_sub = X_train_agg.groupby("installation_id").max()

X_train_sub = X_train_agg.copy()

X_train_sub.reset_index().to_csv("results_in_train_agg.csv",index=False)