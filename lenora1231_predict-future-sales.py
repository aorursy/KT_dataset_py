# Basic packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd # generating random numbers

import datetime # manipulating date formats

# Viz

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots



# Import data



!ls ../input/*

sales=pd.read_csv("../input/sales_train.csv")

item_cat=pd.read_csv("../input/item_categories.csv")

item=pd.read_csv("../input/items.csv")

sub=pd.read_csv("../input/sample_submission.csv")

shops=pd.read_csv("../input/shops.csv")

test=pd.read_csv("../input/test.csv")
print(sales.shape)

sales.head()
print(item_cat.shape)

item_cat.head()
print(item.shape)

item.head()
print(sub.shape)

sub.head()
print(shops.shape)

shops.head()
print(test.shape)

test.head()
#formatting the date column correctly

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# check

print(sales.info())
# new feature shop_item, contact shop_id with item_id

sales['shop_id'].astype(str)

sales['item_id'].astype(str)

sales['shop_item']=sales['shop_id'].astype(str).str.cat(sales['item_id'].astype(str), sep='_')

sales.head()
# let's generate the monthly_sales as the project required

monthly_sales=sales.groupby(by=["shop_item","date_block_num"])[

    "item_cnt_day"].sum()

g=monthly_sales.reset_index()

# transe monthly sales from index to columns 

g=g.pivot(index='shop_item', columns='date_block_num', values='item_cnt_day').reset_index()

g.head()
test['shop_item']=test['shop_id'].astype(str).str.cat(test['item_id'].astype(str), sep='_')

test.head()
# find ID we need to predict sales and merge with history sales

t=test.groupby(by=["shop_item","ID"]).count()

t=t.reset_index()



df = pd.merge(t,g, on=['shop_item'], how='left')



# fill na with 0

df = df.fillna(0)

# drop ID shop_id and item_id

df=df.set_index('ID')

df=df.drop(["shop_item",'shop_id', 'item_id'], axis=1)



df.head()

print(df.describe())
# number of items per category 

x=item.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:10].reset_index()

x

# #plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Top 10 of Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
# the total sales per month for the company

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(12,4))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts)
plt.figure(figsize=(12,4))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
# Extract the training and test data

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape)
# Establish model

model = RandomForestRegressor(n_estimators=10,

                               n_jobs=-1,

                                random_state=42)

model.fit(X_train, y_train)

y_hat=model.predict(X_test)



score=model.score(X_test, y_test)

print(score)


from sklearn.metrics import mean_squared_error



mean_squared_error(y_test, y_hat)
from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout
# Split train and test data

X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

print(X_train.shape, y_train.shape, X_test.shape)
# model training

model_lstm = Sequential()

model_lstm.add(LSTM(15, input_shape=(33,1)))

model_lstm.add(Dense(1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model_lstm.summary()
history = model_lstm.fit(X_train, y_train, validation_data=(X_test, y_test),

                         epochs=32, batch_size=2048, verbose=2, shuffle=True)
# Plot the loss and Accuracay during model training

plt.plot(history.history['loss'], label= 'loss(mse)')

plt.plot(history.history['acc'], label= 'Accuracay')

plt.plot(history.history['val_loss'], label= 'valid loss(mse)')

plt.plot(history.history['val_acc'], label= 'valid Accuracay')

plt.legend(loc=1)
# predict future sales

X_pre=data[:,1:]

y_pre = model_lstm.predict(np.expand_dims(X_pre, axis=2))

# we will keep every value between 0 and 2500

y_pre = y_pre.clip(0,2500)

y_pre
# put data into sub file

df=df.reset_index()

submission = pd.DataFrame({'ID': df['ID'], 'item_cnt_month': y_pre.ravel()})

submission.to_csv('submission.csv',index=False)

submission.head()