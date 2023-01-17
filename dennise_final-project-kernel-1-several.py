# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# For extration of features from text via TD-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
# Load Data from my EDA (https://www.kaggle.com/dennise/coursera-competition-getting-started-eda): 

# (Where I already generated some features like month, meta_category, town and region)

train=pd.read_csv('../input/coursera-competition-getting-started-eda/sales_train.csv')

print(train.shape)



# Aggregate to monthly values

# Groupby all except item_cnt

train=train.groupby(["date_block_num","shop_id","item_id","month","items_english","item_category_id","meta_category","shops_english","town","region"],as_index=False).agg({"item_price":["min","max","mean"],"item_cnt_day":["sum","min","max","mean"]})

print(train.shape)
# Proper naming of columns in one level

train.columns = train.columns.get_level_values(0)

train.columns=["date_block_num","shop_id","item_id","month","items_english","item_category_id","meta_category","shops_english","town","region","item_price_min","item_price_max","item_price_mean","item_cnt_month","item_cnt_day_min","item_cnt_day_max","item_cnt_day_mean"]



# Clip item_cnt to max 20 according to recommendation in course

train["item_cnt_month"].clip(0,20,inplace=True)



train.head()
"""

# Convert to time series, i.e. main identifier is shop_id/item_id-pairs with item_cnt per month from 1-32

# Pivot by month to wide format

train["identifier"]=train["shop_id"].map(str)+"-"+train["item_id"].map(str)

train.set_index("identifier",inplace=True)

"""
final_train=train.pivot_table(index=["shop_id","item_id","items_english","item_category_id","meta_category","shops_english","town","region","item_price_min","item_price_max","item_price_mean","item_cnt_day_min","item_cnt_day_max","item_cnt_day_mean"], columns='date_block_num', values='item_cnt_month',aggfunc='sum').fillna(0.0)

final_train.reset_index(inplace=True)

final_train
"""WORK IN PROGRESS"""

final_train.columns.names = ['']



# Use as index the unique identifier shop-id/item-id (for easier lookup)



final_train["identifier"]=final_train["shop_id"].map(str)+"-"+final_train["item_id"].map(str)

final_train.set_index("identifier",inplace=True)

final_train.reset_index(inplace=True)

final_train
# TD-IF on item and category and shop

    # https://galaxydatatech.com/2018/11/19/feature-extraction-with-tf-idf/

    # https://stackoverflow.com/questions/45961747/append-tfidf-to-pandas-dataframe



# 2 different vectorizations: One for items_english one for shops english



v = TfidfVectorizer()

x = v.fit_transform(final_train['shops_english'])

df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names(), index=final_train.index)



x = v.fit_transform(final_train['items_english'])

df2 = pd.DataFrame(x.toarray(), columns=v.get_feature_names(), index=final_train.index)



final_train =pd.concat([final_train,df1,df2], axis=1)



del df1

del df2



final_train

# Now 253 columns
# Saving the train-dataset to continue in another kernel to not repeat the long-term calculation of features over and over again



# I am sure there is smarter ways to generate these features -> please share with me



final_train.to_csv('final_train_1.csv',index=False)