import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))
train = pd.read_csv('../input/sales_train.csv.gz', compression='gzip')
test = pd.read_csv('../input/test.csv.gz', compression='gzip')
shops = pd.read_csv('../input/shops.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
items = pd.read_csv('../input/items.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv.gz',compression='gzip')
pd.options.display.float_format = '{:20,.2f}'.format
train.describe()
train['date_formatted'] = pd.to_datetime(train.date, format="%d.%m.%Y")
train[['day','month','year']] = train.date.str.split('.',expand=True)
train['sales'] = train.item_price * train.item_cnt_day
fig, axes = plt.subplots(1,4, figsize=(25,5))
train.date_block_num.hist(ax=axes[0])
train[train['item_cnt_day'] < 10].item_cnt_day.hist(ax=axes[1])
np.log(train['item_price']).hist(ax=axes[2])
train.sales.hist(ax=axes[3])

fig, axes = plt.subplots(1,3, figsize=(33,5))

train.year.value_counts().sort_index().plot.bar(ax=axes[0],title='year')
train.month.value_counts().sort_index().plot.bar(ax=axes[1],title='month')
train.day.value_counts().sort_index().plot.bar(ax=axes[2],title='day')
train.groupby('date_formatted').agg({"item_cnt_day": "sum"}).plot(figsize=(15,6),title="Items transacted per day")
fig, axes = plt.subplots(1,2, figsize=(33,5))

train.groupby('year').item_cnt_day.sum().plot.bar(figsize=(15,6),title="Total Items transacted each year", ax=axes[0])
train.groupby('year').item_cnt_day.mean().plot.bar(figsize=(15,6),title="Average item count per transaction each year", ax=axes[1])
train.groupby('date_block_num').agg({"item_cnt_day": "sum"}).plot(figsize=(15,6),title="Items transacted per day")
train['dayofweek'] = train.date_formatted.dt.dayofweek # The day of the week with Monday=0, Sunday=6
train.groupby("dayofweek").agg({"dayofweek": "count"}).plot.bar(figsize=(10, 6));
train['date_month'] = (train.year + train.month)
fig, axes = plt.subplots(1,2, figsize=(25,5))
ax = train.groupby('date_month').item_cnt_day.mean().reset_index().plot(x_compat=True,title="avg item_cnt_day by month", figsize=(20,6), ax=axes[0])
ax.set(xlabel='date month', ylabel='average item_cnt_day')

ax = train.groupby('date_month').item_cnt_day.sum().reset_index().plot(x_compat=True,title="sum item_cnt_day by month", figsize=(20,6), ax=axes[1])
ax.set(xlabel='date month', ylabel='sum item_cnt_day')
plt.scatter(train.index, train.date_block_num)
plt.plot(train.item_cnt_day,'.')
# plot items for each shop -- training set vs test set
fix, axes = plt.subplots(1,2,figsize=(15,3))

train.drop_duplicates(subset=['item_id', 'shop_id']).plot.scatter('item_id', 'shop_id', color='DarkBlue', s = 0.1, ax=axes[0], title="shops vs items train")
test.drop_duplicates(subset=['item_id', 'shop_id']).plot.scatter('item_id', 'shop_id', color='DarkBlue', s = 0.1, ax=axes[1], title="shops vs items test")

