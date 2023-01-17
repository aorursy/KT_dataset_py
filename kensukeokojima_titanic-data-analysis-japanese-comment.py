import numpy as np

import pandas as pd

import os

import sys

import glob

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_table('../input/train.csv',sep=',', index_col=0)   # 学習用データセット

test_df  = pd.read_table('../input/test.csv', sep=',', index_col=0)   # テスト用データセット
'''

基礎統計量出力関数

'''

def df_describe2(input_df):

    # データ型情報

    type_df         = input_df.dtypes.reset_index()

    type_df.columns = ('colname', 'dtype')

    type_df         = type_df.set_index('colname',drop=True)



    # 基礎統計量

    describe_df = input_df.describe(include='all').T

    drop_list   = ['count','unique']

    describe_df = describe_df.drop(drop_list, axis=1)



    # カーディナリティー

    card_df         = input_df.apply(pd.Series.nunique).reset_index()

    card_df.columns = ('colname', 'card')

    card_df         = card_df.set_index('colname',drop=True)

    

    # 行数

    count_df         = input_df.count().reset_index()

    count_df.columns = ('colname', 'count')

    count_df         = count_df.set_index('colname',drop=True)



    # 欠損値数

    null_df         = input_df.isnull().sum().reset_index()

    null_df.columns = ('colname', 'null')

    null_df         = null_df.set_index('colname',drop=True)



    # 結合

    result_df   = pd.concat([type_df, count_df, null_df, card_df, describe_df], axis=1)



    # 結果データフレームを返す

    return result_df
describe_df = df_describe2(train_df)

describe_df
def get_dummy_2(input_df, describe_df, max_card=5):

    tgt_list = describe_df[(describe_df['card']  <= max_card)

                         & (describe_df['dtype'] == 'object')].index.values

    result_df = pd.get_dummies(input_df, columns=tgt_list)

    return result_df
train_df_2  = get_dummy_2(train_df, describe_df, 5)

describe_df = df_describe2(train_df_2)

describe_df
sns.pairplot(train_df_2)
plt.figure(figsize=(12, 12))

sns.heatmap(train_df_2.corr(), annot=True, square=True, vmax=1, vmin=-1, center=0)