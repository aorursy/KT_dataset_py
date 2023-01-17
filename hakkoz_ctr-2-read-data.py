# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import pickle

def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)
first_10k = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv', sep='|', nrows=10000)

first_10k.info()
col_list = first_10k.columns.to_list()
for col in col_list:

    print(col)

    print(first_10k[col].unique())

    print("------")
for col in col_list:

    print(col)

    print(first_10k[col].nunique())

    print("------")
temp_df = first_10k.groupby("uid").count()

temp_df2 = temp_df["creat_type_cd"]



temp_df3 = temp_df2.sort_values(ascending=False)
pd.set_option("display.max_colwidth", 200)



indexes = [1852341, 1738911,   1026288,   1402588,  2135240]

for i in indexes: 

    print(first_10k[first_10k.uid == i][["communication_onlinerate","communication_avgonline_30d"]])
merged_df = pd.DataFrame()

for col in col_list:

    print(col)

    if col != "communication_onlinerate":

        col_df = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv', sep='|', usecols=[col])

        downcasted = pd.to_numeric(col_df[col] , downcast='signed')

    else:

        downcasted = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv', sep='|', usecols=[col], dtype = {col: "category"})

    merged_df=pd.concat([merged_df,downcasted], axis =1)
merged_df.info()
mem_usage(merged_df)
merged_df[merged_df["age"].isna()]
merged_df = merged_df.sample(frac = 1) 

merged_df = merged_df.reset_index(drop=True)



merged_df.head(10)
with open('merged_df.pkl', 'wb') as output:

    pickle.dump(merged_df, output)