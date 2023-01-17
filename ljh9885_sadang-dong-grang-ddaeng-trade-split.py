import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib

matplotlib.rcParams['axes.unicode_minus']=False

plt.style.use('ggplot')

from sklearn.preprocessing import scale,minmax_scale

import os
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary=pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary=summary.reset_index()

    summary['Name']=summary['index']

    summary=summary[['Name','dtypes']]

    summary['Min']=df.min().values

    summary['Max']=df.max().values

    summary['Missing']=df.isnull().sum().values    

    summary['Uniques']=df.nunique().values

    return summary
train_lab=pd.read_csv('../input/bigcontest2019/train_label.csv')

print('train_lab.shape :',train_lab.shape)

train_lab.head()
resumetable(train_lab)
train_tra=pd.read_csv('../input/bigcontest2019/train_trade.csv')

test1_tra=pd.read_csv('../input/bigcontest2019/test1_trade.csv')

test2_tra=pd.read_csv('../input/bigcontest2019/test2_trade.csv')

print('train_tra.shape :',train_tra.shape)

print('test1_tra.shape :',test1_tra.shape)

print('test2_tra.shape :',test2_tra.shape)

train_tra.head()
resumetable(train_tra)
train_sell=train_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

train_buy=train_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터

train_sell.head()
train_buy.head()
train_sell = train_sell.rename(columns = {'source_acc_id':'acc_id',

                                          'source_char_id':'char_id',

                                          'item_type':'sell_item_type',

                                          'item_amount':'sell_item_amount',

                                          'item_price':'sell_item_price',

                                          'time':'sell_time',

                                          'type':'sell_type'})

train_buy = train_buy.rename(columns = {'target_acc_id':'acc_id',

                                        'target_char_id':'char_id',

                                        'item_type':'buy_item_type',

                                        'item_amount':'buy_item_amount',

                                        'item_price':'buy_item_price',

                                        'time':'buy_time',

                                        'type':'buy_type'})

train_sell.head()
train_buy.head()
uniq_acc_id = train_lab['acc_id'].values

print(len(uniq_acc_id))

uniq_acc_id[:10]
train_sell = train_sell[train_sell['acc_id'].isin(uniq_acc_id)]

train_sell.head()
print('판매 데이터의 유니크한 유저아이디 개수 :',train_sell['acc_id'].nunique())
train_sell=train_sell[train_sell['acc_id'].notna()]

print('판매 데이터.shape :',train_sell.shape)

train_sell.head()
train_buy = train_buy[train_buy['acc_id'].isin(uniq_acc_id)]

train_buy.head()
print('구매 데이터의 유니크한 유저아이디 개수 :',train_buy['acc_id'].nunique())
train_buy=train_buy[train_buy['acc_id'].notna()]

print('구매 데이터.shape :',train_buy.shape)

train_buy.head()
l1 = list(train_sell.index)

l2 = list(train_buy.index)
train_sell.loc[train_sell.index.isin(l2),:].head()
train_buy.loc[train_buy.index.isin(l1),:].head()
print('하나의 거래에서 구매자와 판매자가 동시에 label에 있는 유저인 경우 :',train_sell.loc[train_sell.index.isin(l2),:].shape[0])