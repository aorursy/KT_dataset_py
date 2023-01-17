import numpy as np

import pandas as pd

import datetime

import gc

import math

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv", parse_dates = ["first_active_month"])

df_test = pd.read_csv("../input/test.csv", parse_dates = ["first_active_month"])

h_trans = pd.read_csv("../input/historical_transactions.csv", parse_dates = ["purchase_date"])

df_train = df_train[["card_id"]]

df_test = df_test[["card_id"]]
h_trans = h_trans[["card_id", "authorized_flag", "category_1", "category_2", "category_3", "purchase_date", "purchase_amount", "merchant_id"]]

h_trans["authorized_flag"] = h_trans["authorized_flag"].map({"Y":1, "N":0})

h_trans["category_1"] = h_trans["category_1"].map({"Y":1, "N":0})

h_trans = h_trans.fillna(6)
def cat(af, c1, c2, c3):

    s=""

    s += str(int(c2))

    s += str(c1)

    s += str(af)

    s += str(c3)

    

    if s in ["101B", "101A", "611B", "610B", "301B", "501B", "401B", "401A", "301A", "101C", "501A", "100B", "611C","100A", "201A", "610C", "201B", "301C", "300B", "100C", "401C", "601B", "601A"]:

        return s

    else:

        return "0000"



h_trans["cat"] = list(map(cat, h_trans["authorized_flag"], h_trans["category_1"], h_trans["category_2"], h_trans["category_3"]))
#Create more space

h_trans = h_trans.drop(["authorized_flag", "category_1", "category_2", "category_3"], axis=1)
#Code to clean column names

def clean(prefix, df):

    df = df.unstack()

    df.reset_index(inplace=True)

    df.columns = df.columns.droplevel()

    names = []

    i = 0

    for col in df.columns:

        if i == 0:

            names.append("drop")

        elif i == 1:

            names.append("card_id")

        else:

            names.append(prefix+"_"+col)

        i+=1

    df.columns = names

    return df.drop(["drop"],axis=1)
dataframe = h_trans.pivot_table(index='cat', 

                                columns='card_id', 

                                values='merchant_id',

                                fill_value=0, 

                                aggfunc={"count"}).unstack().to_frame().rename(columns={0:"transaction_count"})



dataframe = clean("c", dataframe)

df_train = pd.merge(df_train, dataframe, on = "card_id")

df_test = pd.merge(df_test, dataframe, on = "card_id")



df_train.to_csv("train_counts.csv", index=False)

df_test.to_csv("test_counts.csv", index=False)

df_train = df_train[["card_id"]]

df_test = df_test[["card_id"]]



dataframe = h_trans.pivot_table(index='cat', 

                                columns='card_id', 

                                values='purchase_amount',

                                fill_value=0, 

                                aggfunc={"mean"}).unstack().to_frame().rename(columns={0:"purchase_mean"})



dataframe = clean("pm", dataframe)

df_train = pd.merge(df_train, dataframe, on = "card_id")

df_test = pd.merge(df_test, dataframe, on = "card_id")

df_train.to_csv("train_pm.csv", index=False)

df_test.to_csv("test_pm.csv", index=False)