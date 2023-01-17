# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], date_parser=parser)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

train.head()

test.head()
items.head()
item_cats.head()
shops.head()
# tekrar eden verileri siliyoruz
subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']
print(train.duplicated(subset=subset).value_counts())
train.drop_duplicates(subset=subset, inplace=True)
# test datasında yer almayan dataları eğitimden siliyoruz
test_shops = test.shop_id.unique()
test_items = test.item_id.unique()
train = train[train.shop_id.isin(test_shops)]
train = train[train.item_id.isin(test_items)]

from itertools import product

# dataları birleştiriyoruz
block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), columns=['date_block_num','shop_id'])
shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), columns=['shop_id','item_id'])
all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], how='inner')
print(len(all_combi), 34 * len(test_shops) * len(test_items))

# Aylık veriler için grupluyoruz
train_base = pd.merge(all_combi, train, on=['date_block_num','shop_id','item_id'], how='left')
train_base['item_cnt_day'].fillna(0, inplace=True)
train_grp = train_base.groupby(['date_block_num','shop_id','item_id'])
train_grp.head()
train_monthly = pd.DataFrame(train_grp.agg({'item_cnt_day':['sum','count']})).reset_index()
#print(train_monthly)
train_monthly.columns = ['date_block_num','shop_id','item_id','item_cnt','item_order']
#print(train_monthly[['item_cnt','item_order']].describe())

print(train_monthly.loc[:,['item_cnt']].describe())

# item_cnt -> Gürültü verilerinden kurtuluyoruz.

train_monthly['item_cnt'].clip(0, 20, inplace=True)
print(train_monthly.loc[:,['item_cnt']].describe())


train_monthly.head()

last_train = train_monthly[['date_block_num','shop_id','item_id','item_order']]
X = last_train.iloc[:, :3]
y = last_train.iloc[:, 3:4]

from sklearn.model_selection import train_test_split

# Datayı test ve eğitiö için 2'ye bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
#Model eğitimi
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(8, input_dim=3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


history = model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=64)

_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_pred,y_test)
print('Test Accuracy:', acc*100)
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=10, batch_size=64)
print(history.history)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()
from keras.utils.vis_utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)