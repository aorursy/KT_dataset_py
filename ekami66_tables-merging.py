import sys

import os

import gc

import numpy as np

import pandas as pd

from tqdm import tqdm

from IPython.display import display

import datetime



%matplotlib inline

%config InlineBackend.figure_format = 'retina'
path = "../input/"

files = ["train.csv", "test.csv", "transactions.csv" , "stores.csv", "oil.csv", "items.csv", "holidays_events.csv"]

datasets_path = [path + f for f in files]

print('# File sizes')

for f in datasets_path:

    print(f.ljust(30) + str(round(os.path.getsize(f) / 1000000, 2)) + 'MB')
%%time



# This will take all the data from the test set but not the train set

# It's ~15x less data from the train set 

chunksize = 8_366_470



data_df = {}

for file, path in tqdm(zip(files, datasets_path), total=len(files)):

    name = file.split(".")[0]

    if name == 'train' or name == 'test':

        data_df[name] = pd.read_csv(path, dtype={"date": np.str,

                                                 "id": np.int32,

                                                 "item_nbr": np.int32,

                                                 "onpromotion": np.object,

                                                 "store_nbr": np.int8,

                                                 "unit_sales": np.float32},

                                    parse_dates=["date"], chunksize=chunksize, 

                                    low_memory=False)

        if chunksize:

            data_df[name] = data_df[name].get_chunk()

        data_df[name]["onpromotion"].fillna(False, inplace=True)

        data_df[name]["onpromotion"].map({"True": True, "False": False})

        data_df[name]["onpromotion"].astype(bool)

    elif name == 'transactions':

        data_df[name] = pd.read_csv(path, dtype={"date": np.str,

                                                 "store_nbr": np.int32,

                                                 "transactions": np.int32},

                                   parse_dates=["date"])

    elif name == 'stores':

        data_df[name] = pd.read_csv(path, dtype={"store_nbr": np.int8,

                                                 "city": np.str,

                                                 "state": np.str,

                                                 "type": np.str,

                                                 "cluster": np.int32})

    elif name == 'oil':

        data_df[name] = pd.read_csv(path, dtype={"date": np.str,

                                                 "dcoilwtico": np.float32},

                                   parse_dates=["date"])

    elif name == 'oil':

        data_df[name] = pd.read_csv(path, dtype={"date": np.str,

                                                 "dcoilwtico": np.float32},

                                   parse_dates=["date"])

    elif name == 'items':

        data_df[name] = pd.read_csv(path, dtype={"item_nbr": np.int32,

                                                 "family": np.str,

                                                 "class": np.str,

                                                 "perishable": np.bool})

    elif name == 'holidays_events':

        data_df[name] = pd.read_csv(path, dtype={"date": np.str,

                                                 "type": np.str,

                                                 "locale": np.str,

                                                 "locale_name": np.str,

                                                 "description": np.str,

                                                 "transferred": np.bool},

                                   parse_dates=["date"])

    else:

        data_df[name] = pd.read_csv(path)
for k, df in data_df.items():

    print(k)

    display(df.head())
train_df = data_df["train"]

test_df = data_df["test"]

oil_df = data_df["oil"]

oil_df["date"] = pd.to_datetime(oil_df["date"])

items_df = data_df["items"]

stores_df = data_df["stores"]

transactions_df = data_df["transactions"]

transactions_df["date"] = pd.to_datetime(transactions_df["date"])

holidays_df = data_df["holidays_events"]

holidays_df["date"] = pd.to_datetime(holidays_df["date"])



print("Train set date range: {} to {}".format(train_df["date"].min(), train_df["date"].max()))

print("Test set date range: {} to {}".format(test_df["date"].min(), test_df["date"].max()))
train_df['onpromotion'].fillna(False, inplace=True)

test_df['onpromotion'].fillna(False, inplace=True)



train_date_range = [train_df['date'].min(), train_df['date'].max()]

test_date_range = [test_df['date'].min(), test_df['date'].max()]



merged_df = train_df.append(test_df)

origin_merge_len = len(merged_df)

origin_train_len = len(train_df)

origin_test_len = len(test_df)



assert len(merged_df) == len(train_df) + len(test_df)

print("Merged df size: {}".format(len(merged_df)))

del train_df

del test_df

gc.collect()



display(merged_df.tail())

display(merged_df.dtypes)

print(f"Train set range:{train_date_range}\nTest set range:{test_date_range}")
display(merged_df.isnull().sum().sort_index() / len(merged_df))
merged_df = merged_df.merge(transactions_df, on=["date", "store_nbr"], how='left')

merged_df.head()

print(transactions_df['date'].max())
print(len(merged_df))

display(merged_df.head())

display(merged_df.tail())
stores_df.columns = ['store_' + col if col != "store_nbr" else "store_nbr" for col in stores_df.columns]

stores_df.head(3)
merged_df = merged_df.merge(stores_df, on=["store_nbr"])
print(len(merged_df))

display(merged_df.head())

display(merged_df.tail())
oil_df.tail(3)
merged_df = merged_df.merge(oil_df, on=["date"], how='left')
assert len(merged_df) == origin_merge_len

display(merged_df.head())

display(merged_df.tail())
items_df.columns = ['item_' + col if col != "item_nbr" else "item_nbr" for col in items_df.columns]

items_df.head(3)
merged_df = merged_df.merge(items_df, on=["item_nbr"])
assert len(merged_df) == origin_merge_len

display(merged_df.head())

display(merged_df.tail())
print(f"# of entries: {len(holidays_df)}")

print(holidays_df.nunique())

holidays_df[holidays_df['date'].duplicated(keep=False)].head()
holidays_df = holidays_df[holidays_df["locale_name"] == "Ecuador"]

print(f"# of entries: {len(holidays_df)}")

print(holidays_df.nunique())

holidays_df[holidays_df['date'].duplicated(keep=False)].head()
holidays_df = holidays_df[np.invert(holidays_df['date'].duplicated())]

print(f"# of entries: {len(holidays_df)}")

print(holidays_df.nunique())

holidays_df.head()
holidays_df = holidays_df.drop(["locale", "locale_name"], axis=1)
holidays_df.columns = ['holiday_' + col if col != "date" else "date" for col in holidays_df.columns]

holidays_df.head(3)
holidays_df = holidays_df[holidays_df["holiday_transferred"] != True]

holidays_df = holidays_df.drop(["holiday_transferred"], axis=1)

holidays_df.head()
merged_df = merged_df.merge(holidays_df, on="date", how='left')
assert len(merged_df) == origin_merge_len

display(merged_df.head())

display(merged_df.tail())
print(len(merged_df))

display(merged_df.dtypes)

merged_df.head()
merged_df.set_index("id", inplace=True)
assert len(merged_df) == origin_merge_len
display(merged_df.isnull().sum().sort_index() / len(merged_df))
merged_df = merged_df.drop(["item_nbr", "store_nbr"], axis=1)
train_df = merged_df[(merged_df['date'] >= train_date_range[0]) & (merged_df['date'] <= train_date_range[1])]

train_df = train_df.sort_index()

test_df = merged_df[(merged_df['date'] >= test_date_range[0]) & (merged_df['date'] <= test_date_range[1])]

test_df = test_df.sort_index()

print(f"train_df range: {train_df['date'].min()} to {train_df['date'].max()}")

print(f"test_df range: {test_df['date'].min()} to {test_df['date'].max()}")
assert origin_train_len == len(train_df)

assert origin_test_len == len(test_df)

display(train_df.head(3))

display(test_df.head(3))
# train_df.to_csv("train_joined.csv")

# test_df.to_csv("test_joined.csv")

print("Operation finished!")