

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost

import matplotlib.pyplot as plt

# from sklearn.preprocessing import train_test_split as split

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_path = "../input/competitive-data-science-predict-future-sales/sales_train.csv"

test_path = "../input/competitive-data-science-predict-future-sales/test.csv"

items_path = "../input/competitive-data-science-predict-future-sales/items.csv"

shops_path = "../input/competitive-data-science-predict-future-sales/shops.csv"

item_cat_path = "../input/competitive-data-science-predict-future-sales/item_categories.csv"
items = pd.read_csv(items_path)

cats = pd.read_csv(item_cat_path)

shops = pd.read_csv(shops_path)

train = pd.read_csv( train_path )

test = pd.read_csv( test_path )

plt.figure(figsize = (10,4))

plt.xlim(-100, 3000)

sns.boxplot( x= train.item_cnt_day )

plt.figure( figsize = (10,4) )

plt.xlim(train.item_price.min(), train.item_price.max())

sns.boxplot( x = train.item_price )

plt.show()



print(train.shape[0])

print(len( train[train.item_cnt_day >999 ] ))

print( len(train[ train.item_cnt_day > 500  ]) )

print(len(train[train.item_price >100000 ]))

train = train[(train.item_price < 100000 )& (train.item_cnt_day < 1000)]

print(train.shape[0])
plt.figure(figsize = (10,4))

plt.xlim( -100, 1000 )

sns.boxplot( x= train.item_cnt_day )

plt.figure( figsize = (10,3) )

plt.xlim( train.item_price.min(), train.item_price.max()*1.1 )

sns.boxplot( x = train.item_price )

plt.show()

train = train[train.item_price > 0].reset_index(drop = True)

train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0

shops
train.loc[train.shop_id == 0, "shop_id"] = 57

test.loc[test.shop_id == 0 , "shop_id"] = 57

train.loc[train.shop_id == 1, "shop_id"] = 58

test.loc[test.shop_id == 1 , "shop_id"] = 58

train.loc[train.shop_id == 11, "shop_id"] = 10

test.loc[test.shop_id == 11, "shop_id"] = 10

train.loc[train.shop_id == 40, "shop_id"] = 39

test.loc[test.shop_id == 40, "shop_id"] = 39
shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'

shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )

shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )

shops.loc[shops.city == "!Якутск", "city"] = "Якутск"

category = []

for cat in shops.category.unique():

    print(cat, len(shops[shops.category == cat]) )

    if len(shops[shops.category == cat]) > 4:

        category.append(cat)
shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" )
for cat in shops.category.unique():

    print(cat, len(shops[shops.category == cat]) )
from sklearn.preprocessing import LabelEncoder

shops["shop_category"] = LabelEncoder().fit_transform( shops.category )

shops["shop_city"] = LabelEncoder().fit_transform( shops.city )
shops = shops[["shop_id", "shop_category", "shop_city"]]
cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)

cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "category" ] = "Игры"
cats.shape
category = []

for cat in cats.type_code.unique():

    print(cat, len(cats[cats.type_code == cat]))

    if len(cats[cats.type_code == cat]) > 4: 

        category.append( cat )
cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc")
for cat in cats.type_code.unique():

    print(cat, len(cats[cats.type_code == cat]))
cats.type_code = LabelEncoder().fit_transform(cats.type_code)

cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))

cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] )

cats = cats[["item_category_id", "subtype_code", "type_code"]]
cats.head()
import re

def name_correction(x):

    x = x.lower()

    x = x.partition('[')[0]

    x = x.partition('(')[0]

    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)

    x = x.replace('  ', ' ')

    x = x.strip()

    return x
items = pd.read_csv(items_path)



items["name1"], items["name2"] = items.item_name.str.split("[", 1).str

items["name1"], items["name3"] = items.item_name.str.split("(", 1).str



items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

items = items.fillna('0')



items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")
items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )

items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"

items.loc[ items.type == "", "type"] = "mac"

items.type = items.type.apply( lambda x: x.replace(" ", "") )

items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"

items.loc[ items.type == 'рs3' , "type"] = "ps3"

group_sum = items.groupby(["type"]).agg({"item_id": "count"})

print( group_sum.reset_index() )

group_sum = group_sum.reset_index()
drop_cols = []

for cat in group_sum.type.unique():

#     print(group_sum.loc[(group_sum.type == cat), "item_id"].values[0])

    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:

        drop_cols.append(cat)
drop_cols
items.head()
items.name2 = items.name2.apply( lambda x: "etc" if (x in drop_cols) else x )

items = items.drop(["type"], axis = 1)
items.name2 = LabelEncoder().fit_transform(items.name2)

items.name3 = LabelEncoder().fit_transform(items.name3)



items.drop(["item_name", "name1"],axis = 1, inplace= True)

items.head()
from itertools import product

import time

ts = time.time()

matrix = []

cols  = ["date_block_num", "shop_id", "item_id"]

for i in range(34):

    sales = train[train.date_block_num == i]

    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )



matrix = pd.DataFrame( np.vstack(matrix), columns = cols )

matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)

matrix["shop_id"] = matrix["shop_id"].astype(np.int8)

matrix["item_id"] = matrix["item_id"].astype(np.int16)

matrix.sort_values( cols, inplace = True )

time.time()- ts
train["revenue"] = train["item_cnt_day"] * train["item_price"]
ts = time.time()

group = train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )

group.columns = ["item_cnt_month"]

group.reset_index( inplace = True)

matrix = pd.merge( matrix, group, on = cols, how = "left" )

matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16)

time.time() - ts

test["date_block_num"] = 34

test["date_block_num"] = test["date_block_num"].astype(np.int8)

test["shop_id"] = test.shop_id.astype(np.int8)

test["item_id"] = test.item_id.astype(np.int16)

ts = time.time()



matrix = pd.concat([matrix, test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)

matrix.fillna( 0, inplace = True )

time.time() - ts
ts = time.time()

matrix = pd.merge( matrix, shops, on = ["shop_id"], how = "left" )

matrix = pd.merge(matrix, items, on = ["item_id"], how = "left")

matrix = pd.merge( matrix, cats, on = ["item_category_id"], how = "left" )

matrix["shop_city"] = matrix["shop_city"].astype(np.int8)

matrix["shop_category"] = matrix["shop_category"].astype(np.int8)

matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)

matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8)

matrix["name2"] = matrix["name2"].astype(np.int8)

matrix["name3"] = matrix["name3"].astype(np.int16)

matrix["type_code"] = matrix["type_code"].astype(np.int8)

time.time() - ts
def lag_feature( df,lags, cols ):

    for col in cols:

        print(col)

        tmp = df[["date_block_num", "shop_id","item_id",col ]]

        for i in lags:

            shifted = tmp.copy()

            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]

            shifted.date_block_num = shifted.date_block_num + i

            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
ts = time.time()



matrix = lag_feature( matrix, [1,2,3], ["item_cnt_month"] )

time.time() - ts
ts = time.time()

group = matrix.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_avg_item_cnt"]

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ["date_block_num"], how = "left")

matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float16)

matrix = lag_feature( matrix, [1], ["date_avg_item_cnt"] )

matrix.drop( ["date_avg_item_cnt"], axis = 1, inplace = True )

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_avg_item_cnt' ]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')

matrix.date_item_avg_item_cnt = matrix['date_item_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1,2,3], ['date_item_avg_item_cnt'])

matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = matrix.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_shop_avg_item_cnt"]

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id"], how = "left")

matrix.date_avg_item_cnt = matrix["date_shop_avg_item_cnt"].astype(np.float16)

matrix = lag_feature( matrix, [1,2,3], ["date_shop_avg_item_cnt"] )

matrix.drop( ["date_shop_avg_item_cnt"], axis = 1, inplace = True )

time.time() - ts
ts = time.time()

group = matrix.groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})

group.columns = ["date_shop_item_avg_item_cnt"]

group.reset_index(inplace = True)



matrix = pd.merge(matrix, group, on = ["date_block_num","shop_id","item_id"], how = "left")

matrix.date_avg_item_cnt = matrix["date_shop_item_avg_item_cnt"].astype(np.float16)

matrix = lag_feature( matrix, [1,2,3], ["date_shop_item_avg_item_cnt"] )

matrix.drop( ["date_shop_item_avg_item_cnt"], axis = 1, inplace = True )

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_shop_subtype_avg_item_cnt']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')

matrix.date_shop_subtype_avg_item_cnt = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], ['date_shop_subtype_avg_item_cnt'])

matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'shop_city']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_city_avg_item_cnt']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', "shop_city"], how='left')

matrix.date_city_avg_item_cnt = matrix['date_city_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], ['date_city_avg_item_cnt'])

matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = matrix.groupby(['date_block_num', 'item_id', 'shop_city']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_city_avg_item_cnt' ]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'shop_city'], how='left')

matrix.date_item_city_avg_item_cnt = matrix['date_item_city_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], ['date_item_city_avg_item_cnt'])

matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
ts = time.time()

group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})

group.columns = ["item_avg_item_price"]

group.reset_index(inplace = True)



matrix = matrix.merge( group, on = ["item_id"], how = "left" )

matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)





group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )

group.columns = ["date_item_avg_item_price"]

group.reset_index(inplace = True)



matrix = matrix.merge(group, on = ["date_block_num","item_id"], how = "left")

matrix["date_item_avg_item_price"] = matrix.date_item_avg_item_price.astype(np.float16)

lags = [1, 2, 3]

matrix = lag_feature( matrix, lags, ["date_item_avg_item_price"] )

for i in lags:

    matrix["delta_price_lag_" + str(i) ] = (matrix["date_item_avg_item_price_lag_" + str(i)]- matrix["item_avg_item_price"] )/ matrix["item_avg_item_price"]



def select_trends(row) :

    for i in lags:

        if row["delta_price_lag_" + str(i)]:

            return row["delta_price_lag_" + str(i)]

    return 0



matrix["delta_price_lag"] = matrix.apply(select_trends, axis = 1)

matrix["delta_price_lag"] = matrix.delta_price_lag.astype( np.float16 )

matrix["delta_price_lag"].fillna( 0 ,inplace = True)



features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]

for i in lags:

    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )

    features_to_drop.append("delta_price_lag_" + str(i) )

matrix.drop(features_to_drop, axis = 1, inplace = True)

time.time() - ts

ts = time.time()

group = train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })

group.columns = ["date_shop_revenue"]

group.reset_index(inplace = True)



matrix = matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )

matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)



group = group.groupby(["shop_id"]).agg({ "date_block_num":["mean"] })

group.columns = ["shop_avg_revenue"]

group.reset_index(inplace = True )



matrix = matrix.merge( group, on = ["shop_id"], how = "left" )

matrix["shop_avg_revenue"] = matrix.shop_avg_revenue.astype(np.float32)

matrix["delta_revenue"] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']

matrix["delta_revenue"] = matrix["delta_revenue"]. astype(np.float32)



matrix = lag_feature(matrix, [1], ["delta_revenue"])

matrix["delta_revenue_lag_1"] = matrix["delta_revenue_lag_1"].astype(np.float32)

matrix.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True)

time.time() - ts



matrix
matrix.head().T
matrix["month"] = matrix["date_block_num"] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix["days"] = matrix["month"].map(days).astype(np.int8)
ts = time.time()

matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id","shop_id"])["date_block_num"].transform('min')

matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby(["item_id"])["date_block_num"].transform('min')

time.time() - ts
ts = time.time()

matrix = matrix[matrix["date_block_num"] > 3]

time.time() - ts
import os

import gc

import pickle

import time

import pandas as pd

import numpy as np

from xgboost import XGBRegressor

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
data = matrix.copy()

del matrix

gc.collect()
data[data["date_block_num"]==34].shape
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
X_test.shape
del data

gc.collect();
ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
X_test.shape
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)



# save predictions for an ensemble

pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))

pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))
from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



plot_features(model, (10,14))