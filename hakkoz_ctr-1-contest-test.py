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
first_10k["up_life_duration"]
col_list = first_10k.columns.to_list()
for col in col_list:

    print(col)

    print(first_10k[col].unique())

    print("------")
for col in col_list:

    print(col)

    print(first_10k[col].nunique())

    print("------")
first_10k.groupby("uid").count()["creat_type_cd"].sort_values(ascending=False)
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

        downcasted = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv', sep='|', usecols=[col1], dtype = {col: "category"})

    merged_df=pd.concat([merged_df,downcasted], axis =1)
merged_df.info()
mem_usage(merged_df)
merged_df[merged_df["age"].isna()]
merged_df["age"].isna().sum()
with open('merged_df.pkl', 'wb') as output:

    pickle.dump(merged_df, output)
with open('./merged_df.pkl', 'rb') as inputfile:

    merged_df2 = pickle.load(inputfile)
merged_df2.info()
string_cols.describe()
#read data

train_features_data = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv', sep='|', usecols=["inter_type_cd","career"])

#test_features_data = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/test_data_A.csv', sep='|')



train_features_data
train_features_data["inter_type_cd"]
mem_usage(train_features_data)
train_features_data.nunique()
downcasted = pd.to_numeric(train_features_data["inter_type_cd"], downcast='unsigned')

mem_usage(downcasted)
downcasted
categ = train_features_data["inter_type_cd"].astype('category')

mem_usage(categ)
categ
categ2 =downcasted.astype('category')

mem_usage(categ2)
categ2
train_features_data["career"]
train_features_data["career"].nunique()
mem_usage(train_features_data["career"])
categ3 = train_features_data["career"].astype('category')

mem_usage(categ3)
categ3
strring_col = pd.read_csv('../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv', sep='|', usecols=["communication_onlinerate"])

strring_col
series = strring_col["communication_onlinerate"]
type(series)
strring_col["communication_onlinerate"].nunique()
mem_usage(strring_col["communication_onlinerate"])
categ4 = strring_col["communication_onlinerate"].astype('category')

mem_usage(categ4)
categ4
strring_col["communication_onlinerate"].value_counts()["-1"]