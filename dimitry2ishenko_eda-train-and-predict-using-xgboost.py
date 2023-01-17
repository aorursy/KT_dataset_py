# REQUIREMENTS: these are usually installed in any decent ML environment.

#

#!pip install -U matplotlib numpy pandas tqdm xgboost
import os



in_path = "../input/competitive-data-science-predict-future-sales/"

if os.path.exists(in_path):

    print("Running inside Kaggle")

    tree_method = "hist" # Kaggle GPU VM runs out of RAM



else:

    try:

        from google.colab import drive

        drive.mount("/content/drive")



        os.chdir("/content/drive/My Drive/Colab Notebooks")

        print("Running inside Colab")



        in_path = "./competitive-data-science-predict-future-sales/"

        tree_method = "gpu_hist" # Colab Pro with High-RAM GPU VM



    except:

        in_path = "./"

        tree_method = "gpu_hist"



print("Work dir:", os.getcwd())
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from difflib import SequenceMatcher

from itertools import combinations

from tqdm import tqdm

from xgboost import plot_importance, XGBRegressor



pd.options.display.max_rows = 50

plt.style.use("seaborn")
def compact_types(df):

    for col in df.columns:

        if df[col].dtype == np.float64:

            df[col] = df[col].astype(np.float32)



        elif df[col].dtype == np.int64:

            df[col] = df[col].astype(np.int32)
shops = pd.read_csv(in_path + "shops.csv", header=0, names=["name", "shop"], index_col="shop")

cats  = pd.read_csv(in_path + "item_categories.csv", header=0, names=["name", "cat"], index_col="cat")

items = pd.read_csv(in_path + "items.csv", header=0, names=["name", "item", "cat"], index_col="item")



train = pd.read_csv(in_path + "sales_train.csv", header=0,

    names=["date", "block", "shop", "item", "price", "count"],

    usecols=[1, 2, 3, 4, 5]

)



test = pd.read_csv(in_path + "test.csv", header=0, names=["ID", "shop", "item"])
def get_name(shop): return shops.loc[shop, "name"]

def get_label(shop): return str(shop) + ": " + get_name(shop)



def get_sales(shop):

    sales = train[ train["shop"] == shop ].groupby("block").agg(count=("count", "sum"))

    

    all_blocks = sorted(train["block"].unique())

    sales = sales.reindex(index=all_blocks, fill_value=0)



    return sales.index, sales["count"]



for shop1, shop2 in combinations(shops.index, 2):

    match = SequenceMatcher(a=get_name(shop1), b=get_name(shop2))



    if match.ratio() > .8:

        plt.figure(figsize=(12, 2))



        x1, h1 = get_sales(shop1)

        plt.bar(x=x1, height=h1, width=.5, label=get_label(shop1))

        plt.yticks([])

        plt.legend(loc="upper left")



        x2, h2 = get_sales(shop2)

        plt.bar(x=x2, height=h2, width=.5, label=get_label(shop2), bottom=h1)

        plt.yticks([])

        plt.legend(loc="upper left")
shops_to_replace = { 0: 57, 1: 58, 11: 10, 23: 24, 40: 39 }



train["shop"].replace(shops_to_replace, inplace=True)

test ["shop"].replace(shops_to_replace, inplace=True)



shops.drop(index=shops_to_replace.keys(), inplace=True)
test_shops = test["shop"].unique()



shops_to_delete = []

for shop in shops.index:

    blocks = train[ train["shop"] == shop ]["block"].nunique()



    if blocks < 6:

        plt.figure(figsize=(12, 2))

        x, h = get_sales(shop)



        if shop in test_shops:

            plt.bar(x=x, height=h, width=.5, label=get_label(shop), color="red")

        else:

            plt.bar(x=x, height=h, width=.5, label=get_label(shop))

            shops_to_delete.append(shop)



        plt.yticks([])

        plt.legend(loc="upper left")
print("Deleting shops:", shops_to_delete)



train = train[ ~train["shop"].isin(shops_to_delete) ]

shops.drop(index=shops_to_delete, inplace=True)
cities = shops["name"].str.extract("^([^ ]+)")[0]

shops["city"], cities = cities.factorize()



cities = cities.to_frame(index=False, name="name")

cities.index.name = "city"



print("Extracted", len(cities), "cities")
cats["is_digital"] = cats["name"].str.contains("Цифра")



names = cats["name"].str.replace(" (Цифра)", "", regex=False)

subcats = names.str.extract("^([^-]+)-?(.*)$")



majors = subcats[0].str.strip()

minors = subcats[1].str.strip()

del subcats



# replace empty minors with majors

minors[ minors == "" ] = majors[ minors == "" ]



cats["major"], majors = majors.factorize()

cats["minor"], minors = minors.factorize()



majors = majors.to_frame(index=False, name="name")

majors.index.name = "major"



minors = minors.to_frame(index=False, name="name")

minors.index.name = "minor"



print("Extracted", len(majors), "major and", len(minors), "minor categories")
plt.figure(figsize=(18, 4))

plt.hist(train["price"], log=True, bins=100, color="seagreen", ec="green");
outlier_index = (train["price"] <= 0) | (train["price"] > 49000)

train[outlier_index]
train = train[~outlier_index]



plt.figure(figsize=(18, 4))

plt.hist(train["price"], log=True, bins=100, color="seagreen", ec="green");
plt.figure(figsize=(18, 4))

plt.hist(train["count"], log=True, bins=100, color="teal", ec="darkgreen");
train = train[ (train["count"] > 0) & (train["count"] <= 20) ]



plt.figure(figsize=(18, 4))

plt.hist(train["count"], log=True, bins=100, color="teal", ec="darkgreen")

plt.xticks(range(1, 21));
index_names = ["block", "shop", "item"]

train.set_index(index_names, inplace=True)



train = train.join(shops[["city"]])

train = train.join(items[["cat" ]])

train = train.join(cats [["is_digital", "major", "minor"]], on="cat")

compact_types(train)



train.sort_index(inplace=True)

train
def get_sales(col):

    sales = train.groupby(col).agg(

        count=("count", "sum"),

        count_mean=("count", "mean"),

        price=("price", "mean"),

    )

    sales = sales.add_prefix(col + "_")



    compact_types(sales)

    return sales



blocks = get_sales("block")

shops  =  shops.join(get_sales("shop" ))

cities = cities.join(get_sales("city" )) 

items  =  items.join(get_sales("item" ))

cats   =   cats.join(get_sales("cat"  ))

#majors= majors.join(get_sales("major"))

#minors= minors.join(get_sales("minor"))



compact_types(train)
index_names = ["block", "shop", "item"]



sales_block = train.groupby(index_names).agg(

    block_shop_item_count=("count", "sum"),

    block_shop_item_count_mean=("count", "mean"),

    block_shop_item_price=("price", "mean"),

)



sales_block["block_shop_item_count"] = sales_block["block_shop_item_count"].clip(0, 20)
block_slices = []

for block in tqdm(sales_block.index.unique(0)):

    block_slice = sales_block.loc[block : block]



    shop = block_slice.index.unique(1)

    item = block_slice.index.unique(2)



    block_index = pd.DataFrame(

        index=pd.MultiIndex.from_product([[block], shop, item], names=index_names),

    )

    block_slice = block_index.join(block_slice)



    block_slices.append(block_slice)
test_slice = test[["shop", "item"]]

test_slice["block"] = block + 1

test_slice.set_index(index_names, inplace=True)



block_slices.append(test_slice)



sales_block = pd.concat(block_slices)

del block_slice, block_index, test_slice, block_slices



compact_types(sales_block)

sales_block.info()
sales_block = (sales_block.join(blocks)

    .join( shops.drop(columns="name"))

    .join(cities.drop(columns="name"), on="city")

    .join( items.drop(columns="name"))

    .join(  cats.drop(columns="name"), on="cat")

#   .join(majors.drop(columns="name"), on="major")

#   .join(minors.drop(columns="name"), on="minor")

)



del blocks, shops, cities, items, cats, majors, minors
feat_names = ["shop", "item", "city", "cat"] #, "major", "minor"]

for feat in tqdm(feat_names):



    sales_feat = train.groupby(["block", feat]).agg(

        count=("count", "sum"),

        count_mean=("count", "mean"),

        price=("price", "mean"),

    )

    sales_feat = sales_feat.add_prefix("block_" + feat + "_")



    sales_block = sales_block.join(sales_feat, on=["block", feat])



    # compute price change

    metric = feat + "_price"

    block_price = sales_block["block_" + metric]

    price = sales_block[metric]

    sales_block["block_" + metric + "_change"] = (block_price - price) / price



del sales_feat, block_price, price

del train



compact_types(sales_block)

sales_block.info()
lags = [1, 2, 3, 6, 12]

lag_names = [ name for name in sales_block.columns if name.startswith("block_") ]



for lag in tqdm(lags):



    feat_slice = sales_block[ lag_names ].copy()

    feat_slice.reset_index(inplace=True)



    feat_slice["block"] += lag

        

    # joining on index is much faster

    feat_slice.set_index(index_names, inplace=True)

    feat_slice.rename(columns=lambda name: name + "_lag" + str(lag), inplace=True)



    sales_block = sales_block.loc[ lag : ] # discard "pre-lag" months

    sales_block = sales_block.join(feat_slice)



del feat_slice



compact_types(sales_block)

sales_block.info()
sales_block.drop(columns=[

    name for name in sales_block.columns

        if "block_" in name and "_lag" not in name and name != "block_shop_item_count"

], inplace=True)



sales_block.reset_index(inplace=True)
sales_block["year"] = (sales_block["block"] // 12 + 2013)

sales_block["month"] = (sales_block["block"] % 12 + 1)



dates = sales_block[["year", "month"]].copy()

dates["day"] = 1

sales_block["days"] = pd.to_datetime(dates).dt.daysinmonth



compact_types(sales_block)
sales_block.fillna(0, inplace=True)



for name in tqdm(sales_block.columns):

    if "count" in name and "mean" not in name and "change" not in name:

        if sales_block[name].dtype != np.int32:

            sales_block[name] = sales_block[name].astype(np.int32)



sales_block.to_pickle("sales.pickle")

sales_block.info()
sales_block = pd.read_pickle("sales.pickle")

test = pd.read_csv(in_path + "test.csv", header=0, names=["ID", "shop", "item"])



sales_block = sales_block[[

    "block",

    "shop",

    "item",

    "block_shop_item_count",

    "city",

    "cat",

    "is_digital",

    "major",

    "minor",

    #"year",

    "month",

    "days",



    #"shop_count",

    #"shop_count_mean",

    #"shop_price",

    #

    #"city_count",

    #"city_count_mean",

    #"city_price",

    #

    #"item_count",

    #"item_count_mean",

    #"item_price",

    #

    #"cat_count",

    #"cat_count_mean",

    #"cat_price",

    #

    "block_shop_item_count_lag1",

    #"block_shop_item_count_mean_lag1",

    "block_shop_item_price_lag1",

    #

    #"block_count_lag1",

    "block_count_mean_lag1",

    #"block_price_lag1",

    #

    #"block_shop_count_lag1",

    "block_shop_count_mean_lag1",

    #"block_shop_price_lag1",

    "block_shop_price_change_lag1",

    #

    #"block_item_count_lag1",

    "block_item_count_mean_lag1",

    #"block_item_price_lag1",

    "block_item_price_change_lag1",

    #

    #"block_city_count_lag1",

    "block_city_count_mean_lag1",

    #"block_city_price_lag1",

    #"block_city_price_change_lag1",

    #

    #"block_cat_count_lag1",

    "block_cat_count_mean_lag1",

    #"block_cat_price_lag1",

    #"block_cat_price_change_lag1",

    #

    "block_shop_item_count_lag2",

    #"block_shop_item_count_mean_lag2",

    #"block_shop_item_price_lag2",

    #

    #"block_count_lag2",

    #"block_count_mean_lag2",

    #"block_price_lag2",

    #

    #"block_shop_count_lag2",

    "block_shop_count_mean_lag2",

    #"block_shop_price_lag2",

    #"block_shop_price_change_lag2",

    #

    #"block_item_count_lag2",

    "block_item_count_mean_lag2",

    #"block_item_price_lag2",

    #"block_item_price_change_lag2",

    #

    #"block_city_count_lag2",

    #"block_city_count_mean_lag2",

    #"block_city_price_lag2",

    #"block_city_price_change_lag2",

    #

    #"block_cat_count_lag2",

    #"block_cat_count_mean_lag2",

    #"block_cat_price_lag2",

    #"block_cat_price_change_lag2",

    #

    "block_shop_item_count_lag3",

    #"block_shop_item_count_mean_lag3",

    #"block_shop_item_price_lag3",

    #

    #"block_count_lag3",

    #"block_count_mean_lag3",

    #"block_price_lag3",

    #

    #"block_shop_count_lag3",

    "block_shop_count_mean_lag3",

    #"block_shop_price_lag3",

    #"block_shop_price_change_lag3",

    #

    #"block_item_count_lag3",

    "block_item_count_mean_lag3",

    #"block_item_price_lag3",

    #"block_item_price_change_lag3",

    #

    #"block_city_count_lag3",

    #"block_city_count_mean_lag3",

    #"block_city_price_lag3",

    #"block_city_price_change_lag3",

    #

    #"block_cat_count_lag3",

    #"block_cat_count_mean_lag3",

    #"block_cat_price_lag3",

    #"block_cat_price_change_lag3",

    #

    "block_shop_item_count_lag6",

    #"block_shop_item_count_mean_lag6",

    #"block_shop_item_price_lag6",

    #

    #"block_count_lag6",

    #"block_count_mean_lag6",

    #"block_price_lag6",

    #

    #"block_shop_count_lag6",

    "block_shop_count_mean_lag6",

    #"block_shop_price_lag6",

    #"block_shop_price_change_lag6",

    #

    #"block_item_count_lag6",

    "block_item_count_mean_lag6",

    #"block_item_price_lag6",

    #"block_item_price_change_lag6",

    #

    #"block_city_count_lag6",

    #"block_city_count_mean_lag6",

    #"block_city_price_lag6",

    #"block_city_price_change_lag6",

    #

    #"block_cat_count_lag6",

    #"block_cat_count_mean_lag6",

    #"block_cat_price_lag6",

    #"block_cat_price_change_lag6",

    #

    "block_shop_item_count_lag12",

    #"block_shop_item_count_mean_lag12",

    #"block_shop_item_price_lag12",

    #

    #"block_count_lag12",

    #"block_count_mean_lag12",

    #"block_price_lag12",

    #

    #"block_shop_count_lag12",

    "block_shop_count_mean_lag12",

    #"block_shop_price_lag12",

    #"block_shop_price_change_lag12",

    #

    #"block_item_count_lag12",

    "block_item_count_mean_lag12",

    #"block_item_price_lag12",

    #"block_item_price_change_lag12",

    #

    #"block_city_count_lag12",

    #"block_city_count_mean_lag12",

    #"block_city_price_lag12",

    #"block_city_price_change_lag12",

    #

    #"block_cat_count_lag12",

    #"block_cat_count_mean_lag12",

    #"block_cat_price_lag12",

    #"block_cat_price_change_lag12",

]]

sales_block.info()
test_block = sales_block["block"].max()

valid_block = test_block - 1



X_train = sales_block[ sales_block["block"] <  valid_block ].drop(columns="block_shop_item_count")

y_train = sales_block[ sales_block["block"] <  valid_block ]["block_shop_item_count"]

print("X_train:", X_train.shape, "y_train:", y_train.shape)



X_valid = sales_block[ sales_block["block"] == valid_block ].drop(columns="block_shop_item_count")

y_valid = sales_block[ sales_block["block"] == valid_block ]["block_shop_item_count"]

print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)



X_test = sales_block[ sales_block["block"] == test_block ].drop(columns="block_shop_item_count")

print("X_test :", X_test.shape)



del sales_block
def print_rmse(which, y_true, y_pred):

    print("RMSE", which, np.sqrt( np.square(y_true - y_pred).mean() ))
model_xgb = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42,

    tree_method=tree_method,

    n_jobs=-1,

)

model_xgb.fit(

    X_train, y_train, 

    eval_metric="rmse", 

    eval_set=[(X_valid, y_valid)], 

    verbose=True, 

    early_stopping_rounds=10

)
print_rmse("train", y_train, model_xgb.predict(X_train))

valid_xgb = model_xgb.predict(X_valid)

print_rmse("valid", y_valid, valid_xgb)



test_xgb = model_xgb.predict(X_test)
plt.figure(figsize=(10, 16))

plot_importance(model_xgb, ax=plt.gca())
submission = X_test[["shop", "item"]].copy()

submission["item_cnt_month"] = test_xgb.clip(0, 20)



submission = submission.merge(test, on=["shop", "item"])

submission = submission[["ID", "item_cnt_month"]]



submission.sort_values("ID", inplace=True)
plt.figure(figsize=(12, 4))

plt.hist(submission["item_cnt_month"], bins=105)

submission
submission.to_csv("submission.csv", index=False)
model_xgb.save_model("model_xgb.json")

X_test.to_pickle("X_test.pickle")