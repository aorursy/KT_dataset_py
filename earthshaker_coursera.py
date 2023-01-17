import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/sales-train/sales_train_v2.csv")
df_test = pd.read_csv("../input/exploresales/test.csv")
# df_shopname = pd.read_csv("shops.csv") The name of the shop is not needed, as they are already label encoded
# df_item_cat = pd.read_csv("item_categories.csv") Name of the categories are not needed too
df_items = pd.read_csv("../input/exploresales/items.csv")
df_submit = pd.read_csv("../input/exploresales/submission.csv")
item_dict = df_items[['item_id','item_category_id']].to_dict()
df_train['item_cat_id'] = pd.Series()
df_train['item_cat_id'] = df_train['item_id'].apply(lambda x : item_dict['item_category_id'][x])
len(df_train)
len(df_test)
df_train.head()
items = ['shop_id', 'item_cat_id', 'date_block_num']

for item in items:
    item_counts = df_train[item].value_counts()
    sns.barplot(item_counts.index, item_counts.values)
    plt.title(item+' count')
    plt.show()
df_train.describe()
df_train[df_train['item_price'] < 0].count()
df_train[df_train['item_cnt_day'] < 0].count()
df_train = df_train[(df_train['item_price'] > 0) & (df_train['item_cnt_day'] > 0)]
df_train.head()
dataset = df_train.pivot_table(index=['item_id','shop_id'], columns=['date_block_num'], values=['item_cnt_day'], fill_value=0)
dataset.head()

dataset_filtered = pd.merge(df_test, dataset, on=['item_id', 'shop_id'], how='left')
dataset_filtered.fillna(0, inplace=True)
dataset_filtered.head()
dataset_filtered.drop(['ID', 'shop_id', 'item_id'], axis=1, inplace=True)
X_train = np.expand_dims(dataset_filtered.values[:, :-1], axis=2) # all rows except the last column
y_train = dataset_filtered.values[:, -1:] # last column will be our target value

X_test = np.expand_dims(dataset_filtered.values[:, 1:], axis=2) # shifitng the days by 1, to do a predicting on n+1
model = Sequential()
model.add(GRU(units=128, return_sequences=True,input_shape=(33,1)))
model.add(Dropout(0.3))
model.add(GRU(units=32))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])
model.summary()
reg = model.fit(X_train, y_train, batch_size=512, epochs=10)
LSTM_prediction = model.predict(X_test)
submission = pd.DataFrame({'ID': df_test['ID'], 'item_cnt_month': LSTM_prediction.ravel()})
submission.to_csv('submission.csv',index=False)
len(LSTM_prediction)
