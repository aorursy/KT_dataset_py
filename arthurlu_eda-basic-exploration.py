import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, subprocess
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))
print('# Line count:')
for file in ['sales_train.csv.gz', 'sample_submission.csv.gz', 'test.csv.gz']:
    cmd = "xargs zcat ../input/{} | wc -l".format(file)
    counts = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8')
    print("{}\t{}".format(counts.rstrip(), file))
print('# Line count:')
for file in ['items.csv', 'shops.csv', 'item_categories.csv']:
    lines = subprocess.run(['wc', '-l', '../input/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(lines, end='', flush=True)
fpath = "../input/sales_train.csv.gz"
dtype = {"date_block_num": "int8",
         "item_id": "uint16",
         "shop_id": "int8",
         "item_price": "float64",
         "item_cnt_day": "int16"}
df_Train = pd.read_csv(fpath, compression="gzip", parse_dates=["date"], dtype=dtype,
                       date_parser=lambda x: pd.to_datetime(x, format="%d.%m.%Y"))
df_Train.head()
df_Train.describe()
df_Train.groupby("date").agg({"date": "count"}).plot(figsize=(10, 6));
df_Train.groupby("date_block_num").agg({"date_block_num": "count"}).plot(figsize=(10, 6));
df_Train["month"] = df_Train["date"].dt.month
df_Train.groupby("month").agg({"month": "count"}).plot(figsize=(10, 6));
df_Train["day"] = df_Train["date"].dt.day
df_Train.groupby("day").agg({"day": "count"}).plot(figsize=(10, 6));
df_Train["shop_id"].nunique(), df_Train["item_id"].nunique()
df_Train.groupby("item_id").agg({"shop_id": "nunique"}).reset_index().plot.scatter("item_id", "shop_id", figsize=(10, 6), s=10);
fpath = "../input/test.csv.gz"
dtype = {"item_id": "uint16",
         "shop_id": "int8",}
df_Test = pd.read_csv(fpath, dtype=dtype, index_col="ID")
set(df_Test["shop_id"])-set(df_Train["shop_id"])
df_Test["shop_id"].nunique(), df_Test["item_id"].nunique()
df_Test.groupby("item_id").agg({"shop_id": "nunique"}).reset_index().plot.scatter("item_id", "shop_id", figsize=(10, 6), s=10)
plt.ylim(40, 45);
cols = ["shop_id", "item_id"]
df_ShopItem = df_Train.groupby(cols).agg({"item_id": "nunique"}).rename(columns={"item_id": "Exist"}).reset_index()
df_ShopItem = df_Test.merge(df_ShopItem, on=cols, how="left").fillna(0)
_, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.heatmap(df_ShopItem.pivot("shop_id", "item_id", "Exist"), ax=ax);
df_ShopItem.groupby(["shop_id"]).agg({"Exist": "sum"}).sort_values("Exist")
items_testing = set(df_Test["item_id"])
items_training = set(df_Train["item_id"])
len(items_testing - items_training)
