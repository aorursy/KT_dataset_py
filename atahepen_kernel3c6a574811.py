import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('//kaggle/input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
import matplotlib.pyplot as plt

import seaborn as sns



grouped = train.groupby(["date_block_num"],as_index=False).sum()



fig, ax1 = plt.subplots(figsize=(12,8), dpi= 80)   

obj=sns.barplot(x="date_block_num", y="item_cnt_day",data=grouped, ax=ax1)

obj.set_ylabel("Items Sold")

obj.set_xlabel("Month")
grouped = train.groupby(["date_block_num","item_id"],as_index=False).sum()



my_categories = [6,9,15,16]

for j in my_categories:

    joined = grouped.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()

    joined = joined[joined['item_category_id'] == j]

    avg_categorical_sales = joined.groupby(["date_block_num"],as_index=False).mean()



    unique_items = joined.item_id.unique()





    f=plt.figure(j)

    f.set_figheight(4)

    f.set_figwidth(8)

    ax = plt.gca()

    avg_categorical_sales.plot(kind='line',x='date_block_num',y='item_cnt_day',ax=ax,linewidth=4)

    ax.lines[-1].set_label("Categorical Sales Mean")

    q = 0

    for i in unique_items:

        if q < 10:

            item_holder = joined[joined['item_id'] == i]

            item_holder.plot(kind='line',x='date_block_num',y='item_cnt_day',ax=ax)

            ax.lines[-1].set_label(i)

            q = q+1

    ax.legend()
prepped = train.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)

prepped['total_sales'] = 0



for i in range(0,34):

    prepped['total_sales'] = prepped['total_sales'] + prepped['item_cnt_day',i]



prepped_with_no_zeros = prepped[prepped['total_sales']!= 0]



print("Shape with 0 sales:",prepped.shape)

print("Shape with 0 sales removed:",prepped_with_no_zeros.shape)

prepped.head()
test_prepped = pd.merge(test, prepped, on=['item_id', 'shop_id'], how='left')

test_prepped = test_prepped.fillna(0)

test_prepped = test_prepped.drop(['ID','shop_id','item_id'],axis=1)

# Removing 'total_sales' column

test_prepped = test_prepped.iloc[:,:-1]

test_prepped.head()



prepped = prepped.iloc[:,:-1]

prepped = prepped.reset_index()

prepped = prepped.drop(['shop_id','item_id'],axis=1)

prepped.head()



X_train = np.expand_dims(prepped.values[:, :-1], axis=2)

y_train = prepped.values[:, -1:]



X_test = np.expand_dims(test_prepped.values[:, 1:], axis=2)

print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout



model = Sequential()

model.add(LSTM(units=64, input_shape=(33, 1)))

model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(loss='mse',

              optimizer='adam',

              metrics=['mean_squared_error'])

model.fit(X_train, y_train, batch_size=256, epochs=5)
LSTM_prediction = model.predict(X_test)

LSTM_prediction = np.around(LSTM_prediction)

submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': LSTM_prediction.ravel()})

submission.head(30)

submission.to_csv('submission.csv',index=False)