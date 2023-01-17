# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 200)

pd.set_option('display.width', 320)

# Reading the Data



item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
items.head()
item_categories.head()
sales_train.head()
shops.head()
test.head()
sample_submission.head()
# Descriptive Analysis



# No. of shops

len(sales_train.shop_id.unique())
# No missing values

nfd = sales_train.isnull()

nfd.item_cnt_day.sum().sum()

# EDA

# https://public.tableau.com/profile/caner.irfanoglu#!/vizhome/Kaggle_Predict_Future_Sales/Dashboard1?publish=yes
# Create initial submission with previous values



# Get list of shops

shops = [shop for shop in sorted(list(sales_train.shop_id.unique()))]



# Get list of items

items = [item for item in sorted(list(sales_train.item_id.unique()))]





# Filter df to only test shop & items

test_shops = [shop for shop in sorted(list(test.shop_id.unique()))]

test_items = [item for item in sorted(list(test.item_id.unique()))]

df = sales_train.loc[(sales_train.shop_id.isin(test_shops)) & (sales_train.item_id.isin(test_items))]





pred_df = pd.DataFrame(columns=['ID', 'item_cnt_month'])

# Commented out since nested for loop taking too long

# i = 0 

# for shop in test_shops:

#     print("shop", shop)

#     for item in test_items:

#         i += 1

#         df_filtered = df.loc[(df.item_id == item) & (df.shop_id == shop)]

#         #print('shop = ', shop, 'item = ', item, 'len = ', len(df_filtered))

#         #print(df_filtered)

        

#         # Get ID from testdf

#         test_ID = int(test.loc[(test.shop_id == shop) & (test.item_id == item), 'ID'])



#         if df_filtered.empty:

#             pred_df.loc[i] = [test_ID, 0]

#         else:

#             item_cnt = df_filtered.loc[df_filtered.date_block_num == max(df_filtered.date_block_num), 'item_cnt_day']

#             pred_df.loc[i] = [test_ID, item_cnt.iloc[0]]





pred_df['ID'] = pred_df['ID'].astype(int)



# Writing to csv

#pred_df.to_csv("./submission.csv", index = False)



# Reading from csv

# pred_df = pd.read_csv("../input/futuresalesbenchmark/submission.csv")



# Replicate below https://www.kaggle.com/szhou42/predict-future-sales-top-11-solution



from itertools import product



# For every month we create a grid from all shops/items combinations from that month

grid = []

for block_num in sales_train['date_block_num'].unique():

    cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()

    cur_items = sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

index_cols = ['shop_id', 'item_id', 'date_block_num']

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)



# Aggregations

sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)

groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])

trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()

trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})

trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)



trainset = pd.merge(grid,trainset,how='left',on=index_cols)

trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)



# Get category id

trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')

#trainset.to_csv('trainset_with_grid.csv')

train_subset = trainset.loc[trainset.date_block_num ==max(trainset.date_block_num)]

groups = train_subset[['shop_id', 'item_id', 'item_cnt_month']].groupby(by = ['shop_id', 'item_id'])

train_subset = groups.agg({'item_cnt_month':'sum'}).reset_index()

train_subset.head(3)



merged = test.merge(train_subset, on=["shop_id", "item_id"], how="left")[["ID", "item_cnt_month"]]

merged.isna().sum()



merged['item_cnt_month'] = merged.item_cnt_month.fillna(0).clip(0,20)

submission = merged.set_index('ID')

submission.to_csv('benchmark.csv')