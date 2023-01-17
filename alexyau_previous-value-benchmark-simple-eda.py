import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

items = pd.read_csv('../input/items.csv')

item_cats = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')
train.describe()
test.describe()
submission.describe()
train.head(50).T
train['date'].describe()
'''

train['date'] = pd.to_datetime(train.date)

train = train.sort_values(by='date')

'''
train.tail(50).T
train_oct2015 = train.loc[train['date_block_num'] == 33]
train_oct2015.head()
df_m = train_oct2015.groupby(["shop_id", "item_id"])

month_sum = df_m.aggregate({"item_cnt_day":np.sum}).fillna(0)

month_sum.reset_index(level=["shop_id", "item_id"], inplace=True)

month_sum = month_sum.rename(columns={ month_sum.columns[2]: "item_cnt_month" })

month_sum.describe()
submission.describe()
month_sum['item_id'].value_counts()
test['item_id'].value_counts()
new_submission = pd.merge(month_sum, test, how='right', left_on=['shop_id','item_id'], right_on = ['shop_id','item_id']).fillna(0)

new_submission.drop(['shop_id', 'item_id'], axis=1)

new_submission = new_submission[['ID','item_cnt_month']]
new_submission['item_cnt_month'] = new_submission['item_cnt_month'].clip(0,20)

new_submission.describe()
new_submission.to_csv('previous_value_benchmark.csv', index=False)