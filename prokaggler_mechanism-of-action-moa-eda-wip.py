%reload_ext autoreload

%autoreload 2

%matplotlib inline

from pathlib import Path

import pandas as pd

import seaborn as sns

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="whitegrid")
ROOT = Path("/kaggle/input/lish-moa")

ROOT
def read_csv(ROOT):

    """ Reads all the csv files"""

    data_dict = {}

    print("reading csv files...")

    for file_path in ROOT.glob("*.csv"):

        print(file_path.name)

        df_name =  "df_" + str(file_path.name).split(".")[0]

        data_dict[df_name] = pd.read_csv(file_path)

    print("dataframes created:\n")

    print(data_dict.keys())

    return data_dict

df_dict = read_csv(ROOT)
def print_stats(df_dict):

    """ print df info """

    for df in df_dict.keys():

        print("*"*80)

        print(df)

        print("rows:", df_dict[df].shape[0], "\tcolumns: ", df_dict[df].shape[1])

        print(df_dict[df].info())

        print("missing val", df_dict[df].isna().sum())

print_stats(df_dict)
train_df = df_dict["df_train_features"]

scored_df = df_dict["df_train_targets_scored"]

non_scored_df = df_dict["df_train_targets_nonscored"]
train_df.head(1)
print("number of unique ids:", train_df["sig_id"].nunique())
def cat_bar_plot(train_df=train_df, col=str):

    sns.countplot(data = train_df,x=col)

    plt.show()
cat_bar_plot(col="cp_type")
cat_bar_plot(col="cp_dose")
cat_bar_plot(col="cp_time")