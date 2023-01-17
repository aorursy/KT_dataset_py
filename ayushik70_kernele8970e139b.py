# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/sales_train.csv')

df_train.head(5)
df_item = pd.read_csv('../input/items.csv')

df_item.head(5)
df_category = pd.read_csv('../input/item_categories.csv')

df_category.head(5)
df_shop = pd.read_csv('../input/shops.csv')

df_shop.head(5)
df_test = pd.read_csv('../input/test.csv')

df_test.head(5)
df_sub = pd.read_csv('../input/sample_submission.csv')

df_sub.head(5)


plt.plot(df_train['item_cnt_day'])

plt.title("Number of products sold per day");
sale=df_train.groupby(["date_block_num"])["item_cnt_day"].sum()

sale.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(sale);
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=df_train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(df_train.item_price.min(), df_train.item_price.max()*1.1)

sns.boxplot(x=df_train.item_price)
df_train = df_train[df_train.item_price<100000]

df_train = df_train[df_train.item_cnt_day<1001]
df_train['date'] = pd.to_datetime(df_train['date'], format='%d.%m.%Y')
df_train.head(5)
df_train['date_block_num'].value_counts().sum()
train_dataset = df_train.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num',

                                     fill_value=0)


train_dataset.head(5)
train_dataset = train_dataset.reset_index()

train_dataset.head()

train_dataset = pd.merge(df_test, train_dataset, on=['item_id', 'shop_id'], how='left')

train_dataset.head(5)

train_dataset.isna().sum()
train_dataset=train_dataset.fillna(0)

train_dataset.head(5)

train_dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

train_dataset.head(5)
from sklearn.model_selection import train_test_split
X_train = np.expand_dims(train_dataset.values[:,:-1],axis = 2)

Y_train = train_dataset.values[:,-1:]

X_test = np.expand_dims(train_dataset.values[:,1:],axis = 2)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.10, random_state=1, shuffle=False)

print(X_train.shape,Y_train.shape,X_test.shape,X_valid.shape,Y_valid.shape)
from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
sales_model = Sequential()

sales_model.add(LSTM(units = 64,input_shape = (33,1)))

sales_model.add(Dropout(0.4))

sales_model.add(Dense(1))



sales_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['accuracy'])

sales_model.summary()
sales_model.fit(X_train,Y_train,validation_data=(X_valid, Y_valid),batch_size = 4096,epochs = 10)
model_predict = sales_model.predict(X_test)



model_predict
model_predict = model_predict.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':df_test['ID'],'item_cnt_month':model_predict.ravel()})

submission.head(5)