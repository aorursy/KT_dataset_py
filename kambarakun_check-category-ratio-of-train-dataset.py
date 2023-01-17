import io

import bson

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
bson_train = bson.decode_file_iter(open('../input/train.bson', 'rb'))
dict_bson_train = {}



for idx_bson, data_bson in enumerate(bson_train):

    product_id = data_bson['_id']

    category_id = data_bson['category_id']

    dict_bson_train[product_id] = category_id
df_bson_train = pd.DataFrame.from_dict(dict_bson_train, orient='index')

df_bson_train = df_bson_train.reset_index()

df_bson_train.columns = ['product_id', 'category_id']

df_bson_train.to_csv('df_bson_train.csv')

df_bson_train
df_counts_category_id = pd.DataFrame([df_bson_train.category_id.value_counts()]).T

df_counts_category_id = df_counts_category_id.reset_index()

df_counts_category_id.columns = ['category_id', 'counts']

df_counts_category_id.to_csv('df_counts_category_id.csv')

df_counts_category_id
df_counts_ratio = pd.DataFrame(df_counts_category_id.category_id)

df_counts_ratio['ratio'] = df_counts_category_id.counts / df_counts_category_id.counts.sum()

df_counts_ratio['ratio_sum'] = df_counts_ratio.ratio.cumsum()

df_counts_ratio.to_csv('df_counts_ratio.csv')

df_counts_ratio
df_counts_ratio.plot.bar(x='category_id', y='ratio', figsize=(11.69 * 2, 8.27 * 2))
df_counts_ratio.plot.bar(x='category_id', y='ratio_sum', figsize=(11.69 * 2, 8.27 * 2))