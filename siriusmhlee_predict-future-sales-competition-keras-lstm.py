# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization tool 1

import seaborn as sb # visualization tool 2

import keras as kr # deep learning



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
shop_list = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

item_list = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_category_list = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

train_list = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test_list = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
group_data = train_list.groupby(['date_block_num', 'item_id'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

item_group_data = group_data.groupby(['item_id'])['item_cnt_month'].sum().reset_index()



new_index = (item_group_data['item_cnt_month'].sort_values(ascending=False)).index

item_group_data = item_group_data.reindex(new_index)

sorted_data_top10 = item_group_data.iloc[:10].reset_index()

sorted_data_10to20 = item_group_data.iloc[10:20].reset_index()
plot_data = group_data[group_data['item_id'].isin(sorted_data_top10['item_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.pointplot(x=plot_data['date_block_num'], y=plot_data['item_cnt_month'], hue=plot_data['item_id'], ax=ax)

ax.set_xlabel('Date Block', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('Top10 by Item ID', fontsize=12)

ax.grid(linestyle='-')

plt.show()
plot_data = group_data[group_data['item_id'].isin(sorted_data_10to20['item_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.pointplot(x=plot_data['date_block_num'], y=plot_data['item_cnt_month'], hue=plot_data['item_id'], ax=ax)

ax.set_xlabel('Date Block', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('10to20 by Item ID', fontsize=12)

ax.grid(linestyle='-')

plt.show()
merge_data = pd.merge(train_list[['date_block_num', 'item_id', 'item_cnt_day']], item_list[['item_id', 'item_category_id']], on='item_id')

merge_data.head()



group_data = merge_data.groupby(['date_block_num', 'item_category_id'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

item_category_group_data = group_data.groupby(['item_category_id'])['item_cnt_month'].sum().reset_index()



new_index = (item_category_group_data['item_cnt_month'].sort_values(ascending=False)).index

item_category_group_data = item_category_group_data.reindex(new_index)

sorted_data_top10 = item_category_group_data.iloc[:10].reset_index()

sorted_data_10to20 = item_category_group_data.iloc[10:20].reset_index()
plot_data = group_data[group_data['item_category_id'].isin(sorted_data_top10['item_category_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.pointplot(x=plot_data['date_block_num'], y=plot_data['item_cnt_month'], hue=plot_data['item_category_id'], ax=ax)

ax.set_xlabel('Date Block', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('Top10 by Item Category ID', fontsize=12)

ax.grid(linestyle='-')

plt.show()
plot_data = group_data[group_data['item_category_id'].isin(sorted_data_10to20['item_category_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.pointplot(x=plot_data['date_block_num'], y=plot_data['item_cnt_month'], hue=plot_data['item_category_id'], ax=ax)

ax.set_xlabel('Date Block', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('10to20 by Item Category ID', fontsize=12)

ax.grid(linestyle='-')

plt.show()
group_data = train_list.groupby(['item_id'])['item_cnt_day'].sum().rename('item_cnt_all').reset_index()

group_data2 = train_list.groupby(['item_id'])['item_price'].mean().rename('item_mean_price').reset_index()



merge_data = pd.merge(group_data[['item_id', 'item_cnt_all']], group_data2[['item_id', 'item_mean_price']], on='item_id')
fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.scatterplot(x=merge_data['item_mean_price'], y=merge_data['item_cnt_all'], ax=ax)

ax.set_xlabel('Mean Price', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('Products Sold by Mean Price', fontsize=12)

ax.grid(linestyle='-')

plt.show()
merge_data = merge_data[((merge_data['item_mean_price'] < 100000) & (merge_data['item_cnt_all'] < 25000))]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.scatterplot(x=merge_data['item_mean_price'], y=merge_data['item_cnt_all'], ax=ax)

ax.set_xlabel('Mean Price', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('Products Sold by Mean Price', fontsize=12)

ax.grid(linestyle='-')

plt.show()
new_index = (merge_data['item_cnt_all'].sort_values(ascending=False)).index

merge_data = merge_data.reindex(new_index)

sorted_data_top10 = merge_data.iloc[:10].reset_index()

sorted_data_10to20 = merge_data.iloc[10:20].reset_index()



plot_data = merge_data[merge_data['item_id'].isin(sorted_data_top10['item_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.barplot(x=plot_data['item_mean_price'], y=plot_data['item_cnt_all'], ax=ax)

ax.set_xlabel('Mean Price', fontsize=12)

ax.set_xticklabels(plot_data['item_mean_price'], rotation=45)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('Top10 by Item Mean Price', fontsize=12)

ax.grid(linestyle='-')

plt.show()
group_data = train_list.groupby(['date_block_num', 'shop_id'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

item_group_data = group_data.groupby(['shop_id'])['item_cnt_month'].sum().reset_index()



new_index = (item_group_data['item_cnt_month'].sort_values(ascending=False)).index

item_group_data = item_group_data.reindex(new_index)

sorted_data_top10 = item_group_data.iloc[:10].reset_index()

sorted_data_10to20 = item_group_data.iloc[10:20].reset_index()
plot_data = group_data[group_data['shop_id'].isin(sorted_data_top10['shop_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.pointplot(x=plot_data['date_block_num'], y=plot_data['item_cnt_month'], hue=plot_data['shop_id'], ax=ax)

ax.set_xlabel('Date Block', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('Top10 by Shop ID', fontsize=12)

ax.grid(linestyle='-')

plt.show()
plot_data = group_data[group_data['shop_id'].isin(sorted_data_10to20['shop_id'])]



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.pointplot(x=plot_data['date_block_num'], y=plot_data['item_cnt_month'], hue=plot_data['shop_id'], ax=ax)

ax.set_xlabel('Date Block', fontsize=12)

ax.set_ylabel('Number of Products Sold', fontsize=12)

ax.set_title('10to20 by Shop ID', fontsize=12)

ax.grid(linestyle='-')

plt.show()
#feature - item_cnt_month_0 ~ item_cnt_month_33
group_data = train_list.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()



pivot_data = group_data.pivot_table(index=['shop_id', 'item_id'], columns=['date_block_num'], values=['item_cnt_month'], fill_value=0, aggfunc='sum')

pivot_data = pivot_data.reset_index()

pivot_data['sum_all_month'] = pivot_data.iloc[:, 2:].sum(axis=1)



pivot_data.head()
drop_data = pivot_data[pivot_data['sum_all_month'] > 30]

drop_data = drop_data.drop(['shop_id', 'item_id', 'sum_all_month'], 1)

drop_data = drop_data.clip(0, 20)

drop_data.describe()
train_sequence_data = []

train_sequence_label = []



for row in drop_data.values:

    for start in range(0, 31, 2):

        train_sequence_data.append(np.reshape(row[start:start + 3], (3, 1)))

        train_sequence_label.append(row[start + 3:start + 4])



train_sequence_data = np.array(train_sequence_data)

train_sequence_label = np.array(train_sequence_label)



print(train_sequence_data.shape)

print(train_sequence_label.shape)
model = kr.models.Sequential()

model.add(kr.layers.LSTM(units=128, input_dim=1, input_length=3, return_sequences=True, name='lstm1'))

model.add(kr.layers.LSTM(units=64, input_dim=1, input_length=3, return_sequences=True, name='lstm2'))

model.add(kr.layers.Bidirectional(kr.layers.LSTM(units=64, name='blstm1')))

model.add(kr.layers.Dense(units=32, name='fc1'))

model.add(kr.layers.Dense(units=1, name='fc2'))

model.compile(loss='mse', optimizer=kr.optimizers.Adam(lr=0.00001), metrics=['mean_squared_error'])

model.summary()
fit_history = model.fit(train_sequence_data, train_sequence_label, batch_size=512, epochs=50, verbose=0)
plot_data = pd.DataFrame({'loss':fit_history.history['loss'],

                          'rmse':np.sqrt(fit_history.history['mean_squared_error'])})

plot_data.head()



fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 1, 1)

sb.lineplot(data=plot_data['loss'], ax=ax, label='loss')

sb.lineplot(data=plot_data['rmse'], ax=ax, label='rmse')

ax.set_xlabel('Epochs', fontsize=12)

ax.set_ylabel('Error Value', fontsize=12)

ax.set_title('Error Valu by Epochs', fontsize=12)

ax.grid(linestyle='-')

plt.show()
merge_data = pd.merge(test_list, pivot_data, on=['shop_id', 'item_id'], how='left')

merge_data.fillna(0, inplace=True)

merge_data = merge_data.astype('int64')

merge_data = merge_data.drop([merge_data.columns[0], merge_data.columns[1], merge_data.columns[2], merge_data.columns[37]], 1)



test_sequence_data = []



for row in merge_data.values:

    test_sequence_data.append(np.reshape(row[31:34], (3, 1)))



test_sequence_data = np.array(test_sequence_data)



print(test_sequence_data.shape)
prediction = model.predict(test_sequence_data)
prediction = prediction.clip(0, 20)

submission = pd.DataFrame({'ID':test_list['ID'],

                          'item_cnt_month':prediction.ravel()})



submission.head()
submission.to_csv('submission.csv', index=False)