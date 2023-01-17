import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os, subprocess
import matplotlib.pyplot as plt
from sklearn import preprocessing
fpath = "../input/sales_train.csv.gz"
dtype = {"date_block_num": "int8",
         "item_id": "uint16",
         "shop_id": "int8",
         "item_price": "float64",
         "item_cnt_day": "int16"}
df_Train = pd.read_csv(fpath, compression="gzip", parse_dates=["date"], dtype=dtype,
                       date_parser=lambda x: pd.to_datetime(x, format="%d.%m.%Y"))
fpath = "../input/test.csv.gz"
dtype = {"item_id": "uint16",
         "shop_id": "int8",}
df_Test = pd.read_csv(fpath, dtype=dtype, index_col="ID")
fpath = "../input/items.csv"
dtype = {"item_name": "str",
         "item_id": "uint16",
         "item_category_id": "uint16"}
df_ItemCategory = pd.read_csv(fpath, dtype=dtype)
df_Train = df_Train.merge(df_ItemCategory[["item_id", "item_category_id"]], on="item_id")
df_Test = df_Test.merge(df_ItemCategory[["item_id", "item_category_id"]], on="item_id")
cols = ["shop_id", "item_id"]
df_ShopItem = df_Train.groupby(cols).agg({"item_id": "nunique"}).rename(columns={"item_id": "Exist"}).reset_index()
df_ShopItem = df_Test.merge(df_ShopItem, on=cols, how="left").fillna(0)
df_ShopItem = df_ShopItem.groupby(["shop_id", "item_category_id"]).agg({"Exist": ["sum", "count"]})
df_ShopItem.columns = ["Exist_Sum", "Exist_Count"]
df_ShopItem["Exist_Ratio"] = df_ShopItem["Exist_Sum"] / df_ShopItem["Exist_Count"]
df_ShopItem.reset_index(inplace=True)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.heatmap(df_ShopItem.pivot("shop_id", "item_category_id", "Exist_Sum"), ax=ax);
_, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.heatmap(df_ShopItem.pivot("shop_id", "item_category_id", "Exist_Ratio"), ax=ax);
