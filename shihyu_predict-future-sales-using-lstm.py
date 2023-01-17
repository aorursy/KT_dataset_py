import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import csv
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
train.head()
test.head()
#grouped = pd.DataFrame(train.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index())

#grouped.head()
shop_group = train.groupby('shop_id')

shop_group_list = []

# Every item in the 'shop_group_list' is a shop group

for i in range(len(shop_group)):

    shop_group_list.append(shop_group.get_group(i))



shop_group_list[2].head()
plt.plot(range(34), shop_group_list[35].groupby('date_block_num')['item_cnt_day'].sum())

plt.show()
# Drop not used columns

train = train.drop(['date', 'item_price'], axis=1)

train.head()
# Group by ['date_block_num', 'shop_id', 'item_id']

train = train.groupby(['shop_id', 'item_id', 'date_block_num']).sum().reset_index()

train = train.rename(columns = {"item_cnt_day": "item_cnt_month"})

train.head()
# This cell creates training data. Because it takes a lot of time during commit, so comment it.

'''cols = ['shop_id', 'item_id']

for i in range(29,34):

    cols.append('m'+str(i))

train_months = pd.DataFrame(columns = cols)

test_rows = []



for index, row in test.iterrows():

    temp = []

    sid = row['shop_id']

    iid = row['item_id']

    temp.append(sid)

    temp.append(iid)

    for m in range(29, 34):

        try:

            temp.append(train.loc[(train['shop_id'] == sid) & (train['item_id'] == iid) & (train['date_block_num'] == m)]['item_cnt_month'].values[0])

        except IndexError:

            temp.append(0)    

    test_rows.append(temp)

    if index % 100 == 0:

        print(index)



test_rows = pd.DataFrame(test_rows)

test_rows.columns = cols

train_months = train_months.append(test_rows)

train_months.head()'''
# read the output of above cell

train_new = pd.read_csv('../input/predict-sales-training-data/train_months.csv')
train_new.head()
# Get the categories of testing items

test_cate = []

for index, row in train_new.iterrows():

    test_cate.append(items.loc[(items['item_id'] == row[1])]['item_category_id'].values[0])
cate = pd.DataFrame(test_cate, columns=['item_category'])

train_new = pd.concat([train_new, cate], axis=1)

train_new.head()
x_train = np.expand_dims(np.concatenate((train_new.values[:, :1], train_new.values[:, 2:-2], train_new.values[:, -1:]), axis=1), axis=2)

y_train = train_new.values[:, -2:-1]

x_test = np.expand_dims(np.concatenate((train_new.values[:, :1], train_new.values[:, 3:]), axis=1), axis=2)

print(x_train.shape, y_train.shape, x_test.shape)
train_new.to_csv('train_new_3.csv',index=False)



print('Saved file: ' + 'train_new_3.csv')
from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout
model = Sequential()

model.add(LSTM(units=64, input_shape=(6, 1)))

model.add(Dropout(0.3))

model.add(Dense(1))



model.compile(loss='mse',

              optimizer='adam',

              metrics=['mean_squared_error'])

model.summary()
history = model.fit(x_train, y_train, batch_size=2048, epochs=200)
y_pred = model.predict(x_test)
def write_submission(y_pred):

    with open('submission_file_4.csv', 'w') as output:

        writer = csv.writer(output)

        writer.writerow(['ID', 'item_cnt_month'])

        for i, y in enumerate(y_pred):

            writer.writerow([i, y])

write_submission(y_pred)
'''month_list = [i for i in range(34)]

cnt_month = []

for i in range(34):

    cnt_month.append(0)

def create_month_list(item_id, shop_id):

    shop = []

    for i in range(34):

        shop.append(shop_id)

    item = []

    for i in range(34):

        item.append(item_id)

    months = pd.DataFrame({'shop_id': shop, 'item_id': item, 'date_block_num': month_list, 'item_cnt_month': cnt_month})

    return months'''
'''try:

    train.loc[(train['shop_id'] == 0) & (train['item_id'] == 32) & (train['date_block_num'] == 2)]['item_cnt_month'].values[0]

except IndexError:

    print('error')'''
'''df = pd.DataFrame(columns = ['shop_id', 'item_id', 'date_block_num', 'item_cnt_month'])

pre_pair = ()

for index, row in train.iterrows():

    # If the (item_id, shop_id) is the same as the previous row

    if pre_pair == (row['item_id'], row['shop_id']) and row['item_cnt_month'] != 0:

        temp_index = df.loc[(df['shop_id'] == row['shop_id']) & (df['item_id'] == row['item_id']) & (df['date_block_num'] == row['date_block_num'])].index[0]

        df.loc[temp_index, 'item_cnt_month'] = row['item_cnt_month']

        continue

    pre_pair = (row['item_id'], row['shop_id'])

    df = df.append(create_month_list(row['item_id'], row['shop_id'])).reset_index(drop = True)

    if row['item_cnt_month'] != 0:

        temp_index = df.loc[(df['shop_id'] == row['shop_id']) & (df['item_id'] == row['item_id']) & (df['date_block_num'] == row['date_block_num'])].index[0]

        df.loc[temp_index, 'item_cnt_month'] = row['item_cnt_month']

df

#print(train.loc[(train['shop_id'] == 0) & (train['item_id'] == 30)]['item_cnt_month'].values[0])

#print(train.loc[(train['shop_id'] == 0) & (train['item_id'] == 97)].index[0])'''
'''#train.loc[(train['shop_id'] == 0) & (train['item_id'] == 30)]['item_cnt_month'] = 20

print(train.loc[(train['shop_id'] == 0) & (train['item_id'] == 97)].index[0])'''