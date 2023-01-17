import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

merchants = pd.read_csv('../input/merchants.csv')

new_merchant_t = pd.read_csv('../input/new_merchant_transactions.csv')

his_trans = pd.read_csv('../input/historical_transactions.csv')
print(train.shape)

print(test.shape)

print(merchants.shape)

print(new_merchant_t.shape)

print(his_trans.shape)
train.dtypes
train['first_active_month'] = pd.to_datetime(train['first_active_month']).apply(lambda x: x.strftime('%Y-%m'))
new_merchant_t['city_id'] = new_merchant_t['city_id'].astype(object)

new_merchant_t['merchant_category_id'] = new_merchant_t['merchant_category_id'].astype(object)

new_merchant_t['category_2'] = new_merchant_t['category_2'].astype(object)

new_merchant_t['state_id'] = new_merchant_t['state_id'].astype(object)

new_merchant_t['subsector_id'] = new_merchant_t['subsector_id'].astype(object)

new_merchant_t['purchase_date'] = pd.to_datetime(new_merchant_t['purchase_date'])
his_trans['city_id'] = his_trans['city_id'].astype(object)

his_trans['merchant_category_id'] = his_trans['merchant_category_id'].astype(object)

his_trans['category_2'] = his_trans['category_2'].astype(object)

his_trans['state_id'] = his_trans['state_id'].astype(object)

his_trans['subsector_id'] = his_trans['subsector_id'].astype(object)

his_trans['purchase_date'] = pd.to_datetime(his_trans['purchase_date'])
merchants['merchant_group_id'] = merchants['merchant_group_id'].astype(object)

merchants['merchant_category_id'] = merchants['merchant_category_id'].astype(object)

merchants['subsector_id'] = merchants['subsector_id'].astype(object)

merchants['city_id'] = merchants['city_id'].astype(object)

merchants['state_id'] = merchants['state_id'].astype(object)

merchants['category_2'] = merchants['category_2'].astype(object)
# train, test, ,merchants, new_merchant_t, his_trans

train.columns
test.columns
merchants.columns
new_merchant_t.columns
his_trans.columns