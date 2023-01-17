# General imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os, sys, gc, warnings, random, datetime, math



warnings.filterwarnings('ignore')



from kaggle.competitions import nflrush
# You can only call make_env() once, so don't lose it!

env = nflrush.make_env()
# Configuration

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 100)
SEED = 42

TARGET = 'Yards'
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# You can only iterate through a result from `env.iter_test()` once

# so be careful not to lose it once you start iterating.

iter_test = env.iter_test()
(test_df, sample_prediction_df) = next(iter_test)
train_df.head()
train_df[train_df["PlayId"] == 20170907000118]
train_df.groupby("PlayId").first().head()
train_df_groupby = train_df.groupby("PlayId").first()
print(train_df.shape)

print(train_df_groupby.shape)
train_df.dtypes
#Missing Value Count

train_df.isnull().sum()
test_df.head()
test_df.shape
sample_prediction_df
sample_prediction_df.shape
train_df.columns.tolist()
def make_histogram(col):

    plt.figure(figsize=(15,5))

    plt.subplot(121);

    plt.title(col+"_histogram"+"_Original")

    sns.distplot(train_df[col].dropna(),kde=False);

    plt.subplot(122);

    plt.title(col+"_histogram"+"_groupby")

    sns.distplot(train_df_groupby[col].dropna(),kde=False);
def make_describe(col):

    print(f"###############{col} : describe###############")

    print("\n")

    print(f"######Original Data#######")

    print(train_df[col].describe())

    print("\n")

    print(f"######groupby Data#######")

    print(train_df_groupby[col].describe())
make_histogram("Yards")
make_describe("Yards")
train_df_num = train_df.select_dtypes(exclude="object")



not_num_value = ["GameId","PlayId","Season","Week","Quarter","Down","NflId","NflIdRusher"] 

#I deal with "Season","Week","Quarter","Down","NflId","NflIdRusher" as categorical value. 

train_df_num = train_df_num.drop(not_num_value,axis=1)



not_num_value.remove("PlayId")

train_df_groupby_num = train_df_groupby.drop(not_num_value,axis=1)
train_df_num.describe()
train_df_groupby_num.describe()
train_df_num_list = train_df_num.columns.tolist()



for i in train_df_num_list:

    make_histogram(i)
train_df_cat_list = train_df.select_dtypes("object").columns.tolist()

train_df_cat_list += not_num_value
for i in train_df_cat_list:

    print(f"###############{i} : Value Counts###############")

    print(train_df[i].value_counts())

    print("\n")
for i in train_df_cat_list:

    print(f"###############{i} : Value Counts###############")

    print(train_df_groupby[i].value_counts())

    print("\n")
train_df_cat_list_selected = ['OffenseFormation','PlayDirection','Season','Week','Quarter','Down']
def make_histogram_cat(col):

    plt.figure(figsize=(15,5))

    plt.subplot(121);

    plt.title(col+"_histogram"+"_Original")

    sns.countplot(x=col, data=train_df);

    plt.subplot(122);

    plt.title(col+"_histogram"+"_groupby")

    sns.countplot(x=col, data=train_df_groupby);
for i in train_df_cat_list_selected:

    make_histogram_cat(i)