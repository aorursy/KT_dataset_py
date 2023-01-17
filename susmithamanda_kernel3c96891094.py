# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sales = pd.read_csv("../input/sales_train.csv")

items = pd.read_csv("../input/items.csv")

item_cat = pd.read_csv("../input/item_categories.csv")

shops = pd.read_csv("../input/shops.csv")

test_data = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")

test_data[:6]
df_list = []

index_cols = ['shop_id', 'item_id', 'date_block_num']

df = pd.DataFrame(columns = index_cols)

for each_month in sales['date_block_num'].unique():

    cur_shops = list(sales[sales["date_block_num"]==each_month]['shop_id'].unique())

    for each in cur_shops:

        cur_items = list(sales[sales["date_block_num"] == each_month]['item_id'].unique())

        d = {'shop_id':[each] *len(cur_items) , 'item_id':cur_items, 'date_block_num':[each_month]*len(cur_items)}

    df_each = pd.DataFrame(data=d)

    df_list.append(df_each)

result = pd.concat(df_list)

sales['item_cnt_day'] = sales['item_cnt_day'].clip(0,20)

groups = sales.groupby(index_cols)

trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()

trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
trainset = pd.merge(result,trainset,how='left',on=index_cols)

trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)

trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')

trainset[:6]
train_subset = trainset[trainset.date_block_num == 33]

groups = train_subset[['shop_id', 'item_id', 'item_cnt_month']].groupby(by = ['shop_id', 'item_id'])

train_subset = groups.agg({'item_cnt_month':'sum'}).reset_index()

merged = test_data.merge(train_subset, on=["shop_id", "item_id"], how="left")#[["ID", "item_cnt_month"]] 

merged['item_cnt_month'] = merged.item_cnt_month.fillna(0)#.clip(0,20)

merged[:3]
baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_cnt_month']

train = trainset[baseline_features]

#train = train.set_index('shop_id')

train.loc[:,'item_cnt_month'] = train['item_cnt_month'].astype(int)

train.loc[:,'item_cnt_month'] = train.item_cnt_month.fillna(0)

train[:3]
trainx = train.iloc[:, 0:4]

trainy = train.iloc[:,4]

test_df = pd.DataFrame(test_data, columns = ['shop_id', 'item_id'])

merged_test = pd.merge(test_df, items, on = ['item_id'])[['shop_id','item_id','item_category_id']]

merged_test['date_block_num'] = 33

merged_test[:5]
import xgboost as xgb

model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)

model.fit(trainx, trainy, eval_metric='rmse')

preds = model.predict(merged_test)



df = pd.DataFrame(preds, columns = ['item_cnt_month'])

df