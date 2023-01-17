RETRAIN_FIRST_LEVEL_MODELS = False

RETRAIN_META_MODEL = True
# Numpy and pandas!

import numpy as np

import pandas as pd



# Input files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Python

import pickle

import math

import re

from datetime import datetime

from itertools import product



import matplotlib.pyplot as plt



# ML

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from xgboost import XGBRegressor, plot_importance



pd.set_option('display.float_format', '{:.2f}'.format)
# Import all input CSVs

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

cats = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
#time_group = 'date_block_num'

train['dtime'] = pd.to_datetime(train['date'], format='%d.%m.%Y')



time_group = 'dtime'

# shops_to_plot = [20, 9]  # Weird shops

shops_to_plot = []  # All



to_plot = train.copy()

if shops_to_plot:

    to_plot = to_plot[to_plot.shop_id.isin(shops_to_plot)]



to_plot = to_plot.groupby([time_group, 'shop_id']).item_cnt_day.sum().reset_index()

to_plot.set_index(time_group)

to_plot.sort_index(inplace=True)



fig = plt.figure(figsize = (35, 15))

ax1 = fig.subplots()

for s in to_plot.shop_id.unique():

    shop_plot_X = to_plot[to_plot.shop_id == s][time_group]

    shop_plot_Y = to_plot[to_plot.shop_id == s]['item_cnt_day']

    ax1.plot(shop_plot_X, shop_plot_Y, c=f"C{s}", label=f"Shop {s}")

    ax1.legend()

known_items = train['item_id'].unique()

unknown_items = test[~test.item_id.isin(known_items)]['item_id'].unique()

print(len(known_items))

print(len(unknown_items))
# Outliers

train = train[(train.item_price < 300000) & (train.item_cnt_day < 1000)]

train = train[train.item_price > 0].reset_index(drop = True)



# Duplicated shops

# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11



shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"] = 'СергиевПосад ТЦ "7Я"'

shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )

shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )

shops.loc[shops.city == "!Якутск", "city"] = "Якутск"



# Only keep shop category if there are 5 or more shops of that category, the rest are grouped as "other".

category = []

for cat in shops.category.unique():

    if len(shops[shops.category == cat]) >= 5:

        category.append(cat)

shops.category = shops.category.apply( lambda x: x if (x in category) else "other" )



shops["shop_category"] = LabelEncoder().fit_transform( shops.category )

shops["shop_city"] = LabelEncoder().fit_transform( shops.city )

shops = shops[["shop_id", "shop_category", "shop_city"]]



cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)

cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "category" ] = "Игры"

category = []

for cat in cats.type_code.unique():

    if len(cats[cats.type_code == cat]) >= 5: 

        category.append( cat )

cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc")

cats.type_code = LabelEncoder().fit_transform(cats.type_code)

cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))

cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] )

cats = cats[["item_category_id", "subtype_code", "type_code"]]



def name_correction(x):

    x = x.lower() # all letters lower case

    x = x.partition('[')[0] # partition by square brackets

    x = x.partition('(')[0] # partition by curly brackets

    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters

    x = x.replace('  ', ' ') # replace double spaces with single spaces

    x = x.strip() # remove leading and trailing white space

    return x



# split item names by first bracket

items["name1"], items["name2"] = items.item_name.str.split("[", 1).str

items["name1"], items["name3"] = items.item_name.str.split("(", 1).str



# replace special characters and turn to lower case

items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()



# fill nulls with '0'

items = items.fillna('0')

items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))



# return all characters except the last if name 2 is not "0" - the closing bracket

items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")

items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )

items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"

items.loc[ items.type == "", "type"] = "mac"

items.type = items.type.apply( lambda x: x.replace(" ", "") )

items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"

items.loc[ items.type == 'рs3' , "type"] = "ps3"

group_sum = items.groupby(["type"]).agg({"item_id": "count"})

group_sum = group_sum.reset_index()

drop_cols = []

for cat in group_sum.type.unique():

    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:

        drop_cols.append(cat)

items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x )

items = items.drop(["type"], axis = 1)



items.name2 = LabelEncoder().fit_transform(items.name2)

items.name3 = LabelEncoder().fit_transform(items.name3)



items.drop(["item_name", "name1"],axis=1, inplace=True)
# Let's first create a matrix with combinations of date_block_num, shop_id and item_id for the moments that we have data.

# Observe that if we were to create all possible combinations using:

#     matrix = pd.DataFrame(np.vstack(

#         np.array(list(product(train.date_block_num.unique(), train.shop_id.unique(), train.item_id.unique())), dtype = np.int16)), columns=cols)

# We would end up with a quite sparse 42 260 028 rows matrix, which leads me to overflow problems



# Create matrix with every possible combination with entries for each date_block_num

matrix = []

cols  = ["date_block_num", "shop_id", "item_id"]

for i in range(34):

    sales = train[train.date_block_num == i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype = np.int16))



matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

# 11 128 004 rows matrix created



# Downcast some types to save space

matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)

matrix["shop_id"] = matrix["shop_id"].astype(np.int8)

matrix["item_id"] = matrix["item_id"].astype(np.int16)

matrix.sort_values(cols, inplace = True)



# Create item_cnt_month, our target

group = train.groupby(["date_block_num", "shop_id", "item_id"]).agg({"item_cnt_day": ["sum"]})

group.columns = ["item_cnt_month"]

group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on=cols, how="left")

matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).astype(np.float32)



# Concat test set at the end

test["date_block_num"] = 34

test["date_block_num"] = test["date_block_num"].astype(np.int8)

test["shop_id"] = test.shop_id.astype(np.int8)

test["item_id"] = test.item_id.astype(np.int16)



matrix = pd.concat([matrix, test.drop(["ID"], axis=1)], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace = True)



# Merge other tables into matrix

matrix = pd.merge(matrix, shops, on=["shop_id"], how="left")

matrix = pd.merge(matrix, items, on=["item_id"], how="left")

matrix = pd.merge(matrix, cats, on=["item_category_id"], how="left")

matrix["shop_city"] = matrix["shop_city"].astype(np.int8)

matrix["shop_category"] = matrix["shop_category"].astype(np.int8)

matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)

matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8)

matrix["name2"] = matrix["name2"].astype(np.int8)

matrix["name3"] = matrix["name3"].astype(np.int16)

matrix["type_code"] = matrix["type_code"].astype(np.int8)



# For seasonality

matrix["month"] = matrix["date_block_num"] % 12



# Lag features

def lag_feature(df, lags, cols):

    for col in cols:

        tmp = df[["date_block_num", "shop_id","item_id", col]]

        for i in lags:

            shifted = tmp.copy()

            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_" + str(i)]

            shifted.date_block_num = shifted.date_block_num + i

            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    return df



matrix = lag_feature(matrix, [1, 2, 3], ['item_cnt_month'])



# Trends

matrix['item_cnt_month_1dev_1'] = matrix['item_cnt_month_lag_1'] - matrix['item_cnt_month_lag_2']

matrix['item_cnt_month_1dev_2'] = matrix['item_cnt_month_lag_2'] - matrix['item_cnt_month_lag_3']

matrix['item_cnt_month_2dev'] = matrix['item_cnt_month_1dev_1'] - matrix['item_cnt_month_1dev_2']



# Mean encoding of (item_id, shop_id) tuple

# Here we use the time series approximation where we only make use of known data at each moment (i.e. data with date_block_num value less than the block we are encoding at each moment)

print("Creating mean encoded features. This might take a couple of minutes...")



def add_mean_encoded_feature(df, cols, name):

    groups = []

    for block in df['date_block_num'].unique():

        groupdf = df[df.date_block_num < block].groupby(cols).item_cnt_month.mean().reset_index().rename(columns={'item_cnt_month': name})

        if not groupdf.empty:

            groupdf["date_block_num"] = block

            groups.append(groupdf)

    groupdf = pd.concat(groups, ignore_index=True)



    print(f"Created mean encoded feature {name} for columns: {cols}")

    cols.append("date_block_num")

    return df.merge(groupdf, on=cols, how="left")



matrix = add_mean_encoded_feature(matrix, ["item_id", "shop_id"], "item_shop_menc")

matrix = add_mean_encoded_feature(matrix, ["shop_id"], "shop_menc")

matrix = add_mean_encoded_feature(matrix, ["item_id"], "item_menc")



# Delete first entries as they have less features

n_months_to_delete = 3

matrix = matrix[matrix["date_block_num"] > n_months_to_delete]
matrix.fillna(0, inplace=True)

matrix.reset_index(inplace=True)

print(np.any(np.isnan(matrix)))

print(not np.all(np.isfinite(matrix)))
matrix[['index', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_month',

       'shop_category', 'shop_city', 'item_category_id', 'name2', 'name3']].describe()
matrix[['subtype_code', 'type_code', 'month', 'item_cnt_month_lag_1',

       'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_cnt_month_1dev_1',

       'item_cnt_month_1dev_2', 'item_cnt_month_2dev', 'item_shop_menc',

       'shop_menc', 'item_menc']].describe()
matrix.columns
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

M = 30

V = 33



if RETRAIN_FIRST_LEVEL_MODELS:

    xgbModel = XGBRegressor(

        verbosity=1,

        max_depth=10, #10

        n_estimators=100,

        min_child_weight=0.5, 

        colsample_bytree=0.8, 

        subsample=0.8, 

        eta=0.1, # 0.1

        seed=42,

        reg_lambda=0, # 0

        gamma=0)  # 0



    linModel = LinearRegression()

    nnModel = Pipeline([('scaler', StandardScaler()), ('nn', MLPRegressor(

        hidden_layer_sizes=(100, 50),

        batch_size=100,

        max_iter=20,

        verbose=True

    ))])



    models = {

        'linear': [linModel, {}],

        'xgb': [xgbModel, {

            'eval_metric': "rmse",

            'verbose': True, 

            'early_stopping_rounds': 10

        }],

        'nn': [nnModel, {

            'drop_columns': ['shop_id', 'item_id', 'shop_category', 'shop_city', 'item_category_id', 'name2', 'name3', 'subtype_code', 'type_code']

        }],

    }



    meta_train_data = {}

    meta_valid_data = {}

    meta_test_data = {}

    for block in range(M, 35):

        for k, [m, params] in models.items():

            print(f"Preprocessing for block {block}, {k}")

            drop_columns = params.pop('drop_columns', [])

            drop_columns.append('item_cnt_month')

            X = matrix[matrix.date_block_num < block].drop(drop_columns, axis=1)

            Y = matrix[matrix.date_block_num < block]['item_cnt_month'].clip(0, 20)

            Z = matrix[matrix.date_block_num == block].drop(drop_columns, axis=1)

            ZY = matrix[matrix.date_block_num == block]['item_cnt_month'].clip(0, 20)

            print(f"Fitting block {block}, {k}")

            if k == 'xgb':

                if block < V:

                    params['eval_set'] = [(X, Y), (Z, ZY)]

                else:

                    params['eval_set'] = [(X, Y)]

            m.fit(X, Y, **params)

            print(f"Predicting for block {block}, {k}")

            if block < V:

                meta_train_data.setdefault(k, []).append(m.predict(Z))

            elif block >= V and block != 34:

                meta_valid_data.setdefault(k, []).append(m.predict(Z))

            else:

                meta_test_data.setdefault(k, []).append(m.predict(Z))
meta_train_Y = matrix[(matrix.date_block_num >= M) & (matrix.date_block_num < V)]['item_cnt_month'].clip(0, 20)

meta_valid_Y = matrix[matrix.date_block_num == V]['item_cnt_month'].clip(0, 20)



if RETRAIN_FIRST_LEVEL_MODELS:

    print("Creating datasets")

    meta_train = pd.DataFrame()

    meta_valid = pd.DataFrame()

    meta_test = pd.DataFrame()



    for name, series in meta_train_data.items():

        meta_train[name] = np.concatenate(series)



    for name, series in meta_valid_data.items():

        meta_valid[name] = np.concatenate(series)



    for name, series in meta_test_data.items():

        meta_test[name] = np.concatenate(series)



    for col in meta_train:

        rmse = math.sqrt(mean_squared_error(meta_train_Y, meta_train[col]))

        print(f"{col} rmse: {rmse}")



    print("Saving meta values")

    meta_train.to_csv('meta_train.csv', index=False)

    meta_valid.to_csv('meta_valid.csv', index=False)

    meta_test.to_csv('meta_test.csv', index=False)

else:

    print("Loading meta values")

    meta_train = pd.read_csv("/kaggle/input/meta-models/meta_train.csv")

    meta_valid = pd.read_csv("/kaggle/input/meta-models/meta_valid.csv")

    meta_test = pd.read_csv("/kaggle/input/meta-models/meta_test.csv")
if RETRAIN_META_MODEL:

    if True:

        meta_model = XGBRegressor(

            n_estimators=1000,

            seed=42,

            reg_lambda=0,

            reg_alpha=0)

        meta_model.fit(

                meta_train, 

                meta_train_Y, 

                eval_metric="rmse", 

                eval_set=[(meta_train, meta_train_Y), (meta_valid, meta_valid_Y)], 

                verbose=True, 

                early_stopping_rounds=50)

    else:

        meta_model = LinearRegression()

        meta_model.fit(meta_train, meta_train_Y)



    valid_predicted = meta_model.predict(meta_valid).clip(0, 20)

    rmse = math.sqrt(mean_squared_error(meta_valid_Y, valid_predicted))

    print(f"Validation error of model: {rmse}")

    pickle.dump(meta_model, open("meta_model.p", "wb"))

else:

    print("Loading meta model")

    meta_model = pickle.load(open("meta_model.p", "rb"))

    valid_predicted = meta_model.predict(meta_valid).clip(0, 20)

    rmse  = math.sqrt(mean_squared_error(meta_valid_Y, valid_predicted))

    print(f"Validation error of loaded model: {rmse}")
Y_test = meta_model.predict(meta_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index,

    "item_cnt_month": Y_test

})

submission.to_csv('pablots_submission_multiple.csv', index=False)
# Feature importance

def plot_features(booster, figsize):

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



plot_features(meta_model, (30, 10))