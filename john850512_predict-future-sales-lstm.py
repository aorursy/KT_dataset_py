import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

default_path = '../input/'
!ls ../input
train_df = pd.read_csv(default_path+'sales_train.csv')
items_df = pd.read_csv(default_path+'items.csv')
test_df = pd.read_csv(default_path+'test.csv')
# id對照表..好像用不太到
# item_categories_df = pd.read_csv(default_path+'item_categories.csv')
# shops_df = pd.read_csv(default_path+'shops.csv')
# items_df.drop('item_name', axis=1, inplace=True)
print(train_df.shape, test_df.shape)
train_df.head()
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
# test set中有資料是不在train set的，猜測test set資料室用generator產生的
# 所以那些沒出現過的input就補0就好
dataset = train_df.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)
dataset = dataset.reset_index()
dataset.head()
dataset = pd.merge(test_df, dataset, on=['item_id', 'shop_id'], how='left')
dataset = dataset.fillna(0)
dataset.head()
dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]

X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(units=64, input_shape=(33, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])
model.summary()
history = model.fit(X_train, y_train, batch_size=4096, epochs=10)
plt.plot(history.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(history.history['mean_squared_error']), label= 'rmse')
plt.legend(loc=1)
LSTM_prediction = model.predict(X_test)
LSTM_prediction = LSTM_prediction.clip(0, 20)
submission = pd.DataFrame({'ID': test_df['ID'], 'item_cnt_month': LSTM_prediction.ravel()})
submission.to_csv('submission.csv',index=False)
'''
# check first data in test.csv
check_df = dataset_df[(dataset_df['shop_id']==5) & (dataset_df['item_id']==5037)]
# check_df.plot('date_block_num', 'item_cnt_month')
fig, [ax1, ax2] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='date_block_num', y='item_cnt_month', data=check_df, ax=ax1)
plt.close(g.fig)
g = sns.factorplot(x='date_block_num', y='item_price_month', data=check_df, ax=ax2)
plt.close(g.fig)
'''
