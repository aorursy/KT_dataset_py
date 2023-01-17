# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shops =  pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
items =  pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
items_cat =  pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

train_sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
train_sales['income'] = train_sales['item_cnt_day'] * train_sales['item_price']
train_sales_merged = pd.merge(train_sales, items, on='item_id')

train_sales.head()

train_sales.hist(figsize=(10,10), bins=30)
most_selled_item = train_sales['item_id'].value_counts().index[0]
train_sales_item = train_sales[train_sales['item_id'] == most_selled_item]
train_sales_item.plot(y='item_cnt_day')
train_sales_item['item_cnt_day'].cumsum().plot()
for k in range(5):
    plt.figure()
    k_most_selled_item = train_sales['item_id'].value_counts().index[k]
    train_sales_item = train_sales[train_sales['item_id'] == k_most_selled_item]
    train_sales_item['item_cnt_day'].reset_index().cumsum().plot(y='item_cnt_day')
#target creation
train_sales.groupby(['shop_id', 'date_block_num'])['income'].sum()
def aggregate_sales(gb):
#     gb['item_ids'] = gb['item_id'].tolist()
    gb['avg_item_price'] = gb['item_price'].mean()
    
    gb['total_item_count_month'] = gb['item_cnt_day'].sum()
    gb['total_income_month'] = gb['income'].sum()
    
#     gb['avg_item_count_month'] = gb['item_cnt_day'].mean()
    gb['avg_income_item_type_month'] = gb['income'].mean()
    output = pd.Series([gb['item_id'].tolist(), gb['item_price'].mean(), gb['item_cnt_day'].sum(), gb['income'].sum(), gb['item_cnt_day'].mean(), gb['income'].mean()],
                      index=['item_list', 'avg_item_price', 'total_item_count_month', 'total_income_month', 'avg_item_count_month', 'avg_income_item_type_month'])
    return output

train_sales_grouped = train_sales_merged.drop(['date', 'item_name'], axis=1).groupby(['date_block_num', 'shop_id', 'item_category_id']).apply(aggregate_sales)
train_sales_grouped
train_sales_grouped_processed = train_sales_grouped
train_sales_grouped_processed['item_list'] = train_sales_grouped['item_list'].apply(set).apply(list)
train_sales_grouped_processed
dataset = train_sales_grouped_processed.reset_index()
dataset
dataset['target'] = -1

for i in range(dataset.shape[0]):
    series = dataset.iloc[i]
    month = series['date_block_num']
    shop = series['shop_id']
    item_cat = series['item_category_id']
    mask = (dataset['shop_id'] == shop) & (dataset['date_block_num'] == month+1) & (dataset['item_category_id'] == item_cat)
    try:
        target_count = dataset[mask]['total_item_count_month'].iloc[0]
    except:
        target_count = -1
    dataset.iloc[i,-1] = target_count
    
mask = (dataset['shop_id'] == shop) & (dataset['date_block_num'] == month+1) & (dataset['item_category_id'] == item_cat)
dataset[mask]
dataset_drop = dataset.copy()
columns_to_filter = ['avg_item_price', 'total_item_count_month', 'total_income_month', 'avg_income_item_type_month']
for col in columns_to_filter:
    q_min = dataset_drop[col].quantile(1/100)
    q_max = dataset_drop[col].quantile(0.99)
    mask = (dataset_drop[col] >= q_min) & (dataset_drop[col] <= q_max)
    dataset_drop = dataset_drop[mask]

training_months = list(range(25))
validation_months = list(range(26,34))

train_mask = (dataset_drop['date_block_num'] >= training_months[0]) & (dataset_drop['date_block_num'] <= training_months[-1]) & (dataset_drop['target'] != -1)
val_mask = (dataset_drop['date_block_num'] >= validation_months[1]) & (dataset_drop['target'] != -1)

train_x = dataset_drop[train_mask][['shop_id', 'item_category_id', 'avg_item_price', 'total_item_count_month', 'total_income_month', 'avg_income_item_type_month']]
train_y = dataset_drop[train_mask]['target']

val_x = dataset_drop[val_mask][['shop_id', 'item_category_id', 'avg_item_price', 'total_item_count_month', 'total_income_month', 'avg_income_item_type_month']]
val_y = dataset_drop[val_mask]['target']


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.01, gamma=1)

model.fit(train_x, train_y,
        eval_set=[(val_x, val_y)],
        eval_metric='rmse',
        verbose=True)


from sklearn.metrics import r2_score

y_train_predict = model.predict(train_x)
y_val_predict = model.predict(val_x)

plt.figure()
plt.plot(y_train_predict, train_y, '.')
plt.xlabel('predicted')
plt.ylabel('true')

plt.figure()
plt.plot(y_val_predict, val_y, '.')
plt.title('R2: {}'.format(r2_score(val_y, y_val_predict)))
plt.xlim([-10, 600])
plt.ylim([-10, 600])
plt.xlabel('predicted')
plt.ylabel('true')
