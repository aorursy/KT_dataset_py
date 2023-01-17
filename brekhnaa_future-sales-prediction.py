from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

import pandas as pd

import numpy as np
train=pd.read_csv('../input/sales_train.csv')

print('Training set shape:',train.shape)

#Training set imported
test=pd.read_csv('../input/test.csv')

print('Testing set shape',test.shape)

#testing set imported
items_cats=pd.read_csv('../input/item_categories.csv')

print('Item categories:',items_cats.shape)
items=pd.read_csv('../input/items.csv')

print('Items set shape',items.shape)
shops=pd.read_csv('../input/shops.csv')

print('Shops set shape',shops.shape)
train.columns.values
shops_train=train.groupby(['shop_id']).groups.keys()

len(shops_train)
item_train=train.groupby(['item_id']).groups.keys()

len(item_train)
shops_test=test.groupby(['shop_id']).groups.keys()

len(shops_test)
items_test=test.groupby(['item_id']).groups.keys()

len(items_test)
print('Train DS:',train.columns.values)

print('Test DS:',test.columns.values)

print('Item cats DS:',items_cats.columns.values)

print('Items DS:',items.columns.values)

print('Shops DS:',shops.columns.values)
train.head()
test.head()
train_df=train.groupby(['shop_id','item_id','date_block_num']).sum().reset_index().sort_values(by=['item_id','shop_id'])#.sort_values(by='item_cnt_day',ascending=False)
train_df.head()
train_df['m1']=train_df.groupby(['shop_id','item_id']).item_cnt_day.shift()

train_df['m1'].fillna(0,inplace=True)

train_df
train_df['m2']=train_df.groupby(['shop_id','item_id']).m1.shift()

train_df['m2'].fillna(0,inplace=True)

train_df.head()
train_df.rename(columns={'item_cnt_day':'item_cnt_month'},inplace=True)

train_df.head()
finalDf=train_df[['shop_id','item_id','date_block_num','m1','m2','item_cnt_month']].reset_index()

finalDf.drop(['index'],axis=1,inplace=True)

finalDf.head()
newTest=pd.merge_asof(test, finalDf, left_index=True, right_index=True,on=['shop_id','item_id'])

newTest.head()
model_lstm = Sequential()

model_lstm.add(LSTM(64, input_shape=(1,4)))

model_lstm.add(Dense(1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

y_train=finalDf['item_cnt_month']

newTest.drop(['item_cnt_month'],axis=1,inplace=True)

x_train=finalDf[['shop_id','item_id','m1','m2']]
x_test=newTest[['shop_id','item_id','m1','m2']]

x_test.shape
x_test_reshaped=x_test.values.reshape((x_test.values.shape[0], 1, x_test.values.shape[1]))

x_test_reshaped.shape
history = model_lstm.fit(x_train_reshaped, y_train, epochs=20, batch_size=100, shuffle=False)

#On my laptop i used 100 epochs and 10 batch size but it was taking too much time on Kaggle to run so i changed the parameters

y_pre = model_lstm.predict(x_test_reshaped)

y_preF=np.round(y_pre,0)

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':y_preF.ravel()})

#submission.to_csv('submission_Fsales.csv',index = False)