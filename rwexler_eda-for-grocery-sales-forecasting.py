# import necessary modules

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import gc
# read data sets (this may take a few minutes)

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_sub = pd.read_csv("../input/sample_submission.csv")

df_stores = pd.read_csv("../input/stores.csv")

df_items = pd.read_csv("../input/items.csv")

df_trans = pd.read_csv("../input/transactions.csv")

df_oil = pd.read_csv("../input/oil.csv")

df_holiday = pd.read_csv("../input/holidays_events.csv")
# inspect training set

print(df_train.shape)

df_train.head()
# convert date to datetime

df_train["date"] =  pd.to_datetime(df_train["date"])
df_train["date"].dt.year.value_counts(sort = False).plot.bar()
df_train_2016 = df_train[df_train["date"].dt.year == 2016]

del df_train; gc.collect() # free up some memory
df_train_2016["date"].dt.month.value_counts(sort = False).plot.bar()
df_train_2016["date"].dt.day.value_counts(sort = False).plot.bar()
df_train_2016["store_nbr"].unique()
df_train_2016["store_nbr"].value_counts(sort = False).plot.bar()
df_train_2016["item_nbr"].unique().shape[0]
stores = np.arange(1, 55)

items_store = np.zeros((54, ))

for i, store in enumerate(stores) :

    items_store[i] = df_train_2016["item_nbr"][df_train_2016["store_nbr"] \

                                               == store].unique().shape[0]

sns.barplot(stores, items_store)
df_train_2016["unit_sales"].describe()
df_train_2016["onpromotion"].value_counts()
3514584 / (3514584 + 31715287) * 100
df_train_2016.isnull().sum()
unit_sales = df_train_2016["unit_sales"].values

del df_train_2016; gc.collect()
plt.scatter(x = range(unit_sales.shape[0]), y = np.sort(unit_sales))
del unit_sales; gc.collect()
# inspect test set

print(df_test.shape)

df_test.head()
# convert date to datetime

df_test["date"] =  pd.to_datetime(df_test["date"])

df_test["date"].dt.year.value_counts(sort = False).plot.bar()
df_test["date"].dt.month.value_counts(sort = False).plot.bar()
df_test["date"].dt.day.value_counts(sort = False).plot.bar()
df_test["store_nbr"].value_counts(sort = False).plot.bar()
df_test["item_nbr"].unique().shape[0]
stores = np.arange(1, 55)

items_store = np.zeros((54, ))

for i, store in enumerate(stores) :

    items_store[i] = df_test["item_nbr"][df_test["store_nbr"] \

                                         == store].unique().shape[0]

sns.barplot(stores, items_store)
df_test["onpromotion"].value_counts()
198597/(198597 + 3171867) * 100
print(df_sub.shape)

df_sub.head()
(df_sub["id"] - df_test["id"]).sum()
del df_test, df_sub; gc.collect()
print(df_stores.shape)

df_stores.head()
df_stores.isnull().sum()
df_stores["city"].unique().shape[0]
df_stores["city"].value_counts(sort = False).plot.bar()
print(df_stores["state"].unique().shape[0])

df_stores["state"].value_counts(sort = False).plot.bar()
print(df_stores["type"].unique().shape[0])

df_stores["type"].value_counts(sort = False).plot.bar()
print(df_stores["cluster"].unique().shape[0])

df_stores["cluster"].value_counts(sort = False).plot.bar()
df_stores.groupby(["type", "cluster"]).size()
df_stores.groupby(["type", "state"]).size()
del df_stores; gc.collect()
print(df_items.shape)

df_items.head()
print(df_items["family"].unique().shape[0])

df_items["family"].value_counts(sort = False).plot.bar()
print(df_items["class"].unique().shape[0])

print(df_items["class"].value_counts()[0:5])

df_items["class"].plot.hist(bins = 50)
df_items["perishable"].value_counts()
986 / (986 + 3114) * 100
del df_items; gc.collect()
print(df_trans.shape)

print(df_trans.head())

df_trans.isnull().sum()
df_trans["date"] =  pd.to_datetime(df_trans["date"])
df_trans["date"].dt.year.value_counts(sort = False).plot.bar()
df_trans["date"].dt.month.value_counts(sort = False).plot.bar()
df_trans["date"].dt.day.value_counts(sort = False).plot.bar()
df_trans["store_nbr"].value_counts(sort = False).plot.bar()
df_trans["transactions"].plot.hist(bins = 100)
del df_trans; gc.collect()