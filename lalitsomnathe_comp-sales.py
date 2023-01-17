# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
    
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
sales_train=pd.read_csv("../input/sales_train.csv")
display(sales_train.head())
sales_train['date']=pd.to_datetime(sales_train['date'], format= "%d.%m.%Y")
sales_train.head()
print(sales_train['shop_id'].nunique())
print(sales_train['item_id'].nunique())
sales_train.describe(include='all')
sales_train.info()
items=pd.read_csv("../input/items.csv")
items.head()
items.info()
shops=pd.read_csv("../input/shops.csv")
shops.head()
shops.info()
item_cat=pd.read_csv("../input/item_categories.csv")
item_cat.head()
sales_train['mon']=sales_train['date'].dt.month
sales_train['year']=sales_train['date'].dt.year
sales_train.head()
temp_sales=sales_train.head(100)
temp_sales.info()
sales_new_train=pd.merge(sales_train, items, on='item_id', how='inner')
type(sales_new_train)
sales_new_train.drop(['item_name','mon'], axis=1, inplace=True)
sales_new_train.head()
sales_new_train.set_index
print('reseted index')
sales_new_train=sales_new_train.sort_values(by=['date'])
sales_new_train.set_index('date',inplace=True)
sales_new_train.head()
sales_temp=sales_new_train['item_cnt_day'].groupby([sales_new_train['shop_id'],sales_new_train['item_id'],sales_new_train['date_block_num'],sales_new_train['year'],sales_new_train['item_price']]).sum()
# .sort_values(by=['mon','year','shop_id','item_id'])
# sal=sales_new_train['item_cnt_day'].groupby(sales_new_train['shop_id'],sales_new_train['item_id'],sales_new_train['date_block_num']
sales_temp=sales_temp.reset_index()
sales_temp.head()
sales_temp1=pd.DataFrame(sales_temp).sort_values(by=['shop_id','item_id','date_block_num','year'])
# sales_temp1[(((sales_temp1['shop_id']==35) & (sales_temp1['item_id']==31)) |((sales_temp1['shop_id']==2) & (sales_temp1['item_id']==1121)) )
#             & (sales_temp1['date_block_num']==1)]
# # ['item_price'].mean()
t1=sales_temp1[((sales_temp1['shop_id']==35) & (sales_temp1['item_id']==31)) | ((sales_temp1['shop_id']==2) & (sales_temp1['item_id']==1121))]
t1
# sales_temp1['item_price'].groupby(sales_temp1['date_block_num']).mean()
# t1=sales_temp1['item_price'].groupby([sales_temp1['shop_id'],sales_temp1['item_id'],sales_temp1['date_block_num'],sales_temp1['year']]).mean()
t2=t1['item_price'].groupby([t1['shop_id'],t1['item_id'],t1['date_block_num'],t1['year']]).mean()
t2=t2.reset_index()
t2
t3=pd.merge(t1,t2, on=['shop_id','item_id','date_block_num','year'])
t3
t3['item_cnt_day'].groupby([t3['shop_id'],t3['item_id'],t3['date_block_num'],t3['year'],t3['item_price_y']]).sum()

sales_train[(sales_train['shop_id']==35) & (sales_train['item_id']==31) ]#& (sales_train['date_block_num']==1)]
# sales_train[(sales_train['shop_id']==2) & (sales_train['item_id']==1121) ]
sales_temp2=sales_temp1['item_price'].groupby([sales_temp1['shop_id'],sales_temp1['item_id'],sales_temp1['date_block_num'],sales_temp1['year']]).mean()
sales_temp2=sales_temp2.reset_index()
sales_temp2.head()
sales_temp3=pd.merge(sales_temp1,sales_temp2, on=['shop_id','item_id','date_block_num','year'])
sales_temp3.head()
sales_temp4=sales_temp3['item_cnt_day'].groupby([sales_temp3['shop_id'],sales_temp3['item_id'],sales_temp3['date_block_num'],sales_temp3['year'],sales_temp3['item_price_y']]).sum()
sales_temp4.head()
sales_train[(sales_train['shop_id']==0) & (sales_train['item_id']==32 )]
sales_temp4=sales_temp4.reset_index()
sales_temp4.head()
sales_temp5=pd.merge(sales_temp4, items, on='item_id', how='inner')
sales_temp5.drop(['item_name'], axis=1, inplace=True)
sales_temp5=sales_temp5[['shop_id','item_id','date_block_num','year','item_category_id','item_price_y','item_cnt_day']]
sales_temp5=sales_temp5.rename(columns={'item_cnt_day':'cur_items_per_month' })
sales_temp5.head()
sales_temp5[(sales_temp5['shop_id']==35) & (sales_temp5['item_id']==31) ]
from sklearn.preprocessing import StandardScaler
scaler_price=StandardScaler()
scaler_cnt=StandardScaler()

sales_temp5["cur_avg_prc"]=scaler_price.fit_transform(sales_temp5["item_price_y"].values.reshape(-1,1))
sales_temp5["cur_mon_cnt"]=scaler_cnt.fit_transform(sales_temp5["cur_items_per_month"].values.reshape(-1,1))
sales_temp5.head()
sales_temp5.drop(["item_price_y","cur_items_per_month"], axis=1, inplace =True)
sales_temp5.head()
sales_temp5["last_price"]=sales_temp5.groupby(["item_id","shop_id"]).cur_avg_prc.shift().fillna(sales_temp5["cur_avg_prc"])
sales_temp5["last_cnt"]=sales_temp5.groupby(["item_id","shop_id"]).cur_mon_cnt.shift().fillna(sales_temp5["cur_mon_cnt"])
# sales_temp5.head()
sales_temp5[(sales_temp5['shop_id']==35) & (sales_temp5['item_id']==31) ]
# temp_new["last_price1"]=temp_new.groupby(["item_id","shop_id"]).last_price.shift().fillna(temp_new["last_price"])
# temp_new["last_cnt1"]=temp_new.groupby(["item_id","shop_id"]).last_cnt.shift().fillna(temp_new["last_cnt"])
# temp_new[-10:]
sales_temp5["last_price1"]=sales_temp5.groupby(["item_id","shop_id"]).last_price.shift().fillna(sales_temp5["last_price"])
sales_temp5["last_cnt1"]=sales_temp5.groupby(["item_id","shop_id"]).last_cnt.shift().fillna(sales_temp5["last_cnt"])
# sales_temp5.head()
sales_temp5[(sales_temp5['shop_id']==35) & (sales_temp5['item_id']==31) ]
sales_temp5["last_price2"]=sales_temp5.groupby(["item_id","shop_id"]).last_price1.shift().fillna(sales_temp5["last_price1"])
sales_temp5["last_cnt2"]=sales_temp5.groupby(["item_id","shop_id"]).last_cnt1.shift().fillna(sales_temp5["last_cnt1"])

sales_temp5["last_price3"]=sales_temp5.groupby(["item_id","shop_id"]).last_price2.shift().fillna(sales_temp5["last_price2"])
sales_temp5["last_cnt3"]=sales_temp5.groupby(["item_id","shop_id"]).last_cnt2.shift().fillna(sales_temp5["last_cnt2"])

sales_temp5["last_price4"]=sales_temp5.groupby(["item_id","shop_id"]).last_price3.shift().fillna(sales_temp5["last_price3"])
sales_temp5["last_cnt4"]=sales_temp5.groupby(["item_id","shop_id"]).last_cnt3.shift().fillna(sales_temp5["last_cnt3"])

sales_temp5["last_price5"]=sales_temp5.groupby(["item_id","shop_id"]).last_price4.shift().fillna(sales_temp5["last_price4"])
sales_temp5["last_cnt5"]=sales_temp5.groupby(["item_id","shop_id"]).last_cnt4.shift().fillna(sales_temp5["last_cnt4"])
# sales_temp5.head()
sales_temp5[(sales_temp5['shop_id']==35) & (sales_temp5['item_id']==31) ]
# sales_temp6=sales_temp5[['year','shop_id','item_id','date_block_num','item_category_id','last_price', 'last_cnt','last_price1', 'last_cnt1','cur_avg_prc','cur_mon_cnt']]
# # sales_temp6=sales_temp6.rename(columns={'new_prc': 'cur_prc', 'new_cnt': 'cur_cnt'})
# sales_temp6.head()
sales_temp5.drop(['cur_avg_prc',], axis=1, inplace=True)
sales_temp.head()
sales_temp5[(sales_temp5['shop_id']==35) & (sales_temp5['item_id']==31) ]
# sales_temp6.info()
col=sales_temp5.columns
col=(list(col))
# sales_temp6=sales_temp5[['year','shop_id','item_id','date_block_num','item_category_id','last_price', 'last_cnt','last_price1', 'last_cnt1','cur_avg_prc','cur_mon_cnt']]
col1=[]
for i in col:
    if i.startswith('last'):
        col1.append(i)
for i in ('shop_id','item_id','date_block_num','year','item_category_id','cur_mon_cnt'):
    col1.append(i)
print(len(col1))

sales_temp6=sales_temp5[col1]
sales_temp6[(sales_temp6['shop_id']==35) & (sales_temp6['item_id']==31) ]
block=sales_temp6[(sales_temp6["year"]==sales_temp6["year"].max())]["date_block_num"].max()
print ( 'max block =', block)
# idx=sales_temp6.index[(sales_temp6["year"]==sales_temp6["year"].max())  & (sales_temp6["date_block_num"]==33)]#& (sales_temp4["mon"]==10)
# sales_temp6.loc[idx].count() #test == 33198 total 1739021  31531
# sales_temp7= sales_temp6.drop(sales_temp6.index[idx])
# print(sales_temp7.count()) # train 1705824 1577593
sales_temp7=sales_temp6.copy()
idx=sales_temp7.index[(sales_temp7["year"]==sales_temp7["year"].max())  & (sales_temp7["date_block_num"]==33)]#& (sales_temp4["mon"]==10)
sales_temp7=sales_temp7.drop(['date_block_num','year','item_category_id'], axis=1)
test=sales_temp7.loc[idx].values

sales_temp7= sales_temp7.drop(sales_temp7.index[idx])

train=sales_temp7.values
# sales_vals=sales_temp3.values
# train=sales_temp7.values
# test=sales_temp6.loc[idx].values
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
import matplotlib.pyplot

model = Sequential()
model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.4)) #loss: 0.9974 - val_loss: 2.8298
model.add(Dense(1,activation='tanh'))
opt=RMSprop(lr=0.0017,decay=0.02)
#opt=Adam(lr)
model.compile(loss='mean_squared_error', optimizer=opt)
# fit network
history = model.fit(train_X, train_y, epochs=40, batch_size=256, validation_data=(test_X,test_y), verbose=2, shuffle=False)
# plot history

from  matplotlib import pyplot 
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# sales_temp6.head()
# sales_temp4.head()

sales_temp_new=pd.merge(sales_temp4, items, on='item_id', how='inner')
sales_temp_new.drop(['item_name'], axis=1, inplace=True)
sales_temp_new=sales_temp_new[['shop_id','item_id','date_block_num','year','item_category_id','item_price_y','item_cnt_day']]
sales_temp_new=sales_temp_new.rename(columns={'item_cnt_day':'cur_items_per_month' })
sales_temp_new.head()


from sklearn.preprocessing import StandardScaler
scaler_price1=StandardScaler()
scaler_cnt1=StandardScaler()

sales_temp_new["cur_avg_prc"]=scaler_price1.fit_transform(sales_temp_new["item_price_y"].values.reshape(-1,1))
sales_temp_new["cur_mon_cnt"]=scaler_cnt1.fit_transform(sales_temp_new["cur_items_per_month"].values.reshape(-1,1))
sales_temp_new.drop(['item_price_y','cur_items_per_month'],axis=1,inplace=True)
sales_temp_new.head()
sales_temp_new1=pd.DataFrame()
sales_temp_new1['last_prc']=sales_temp_new.groupby(['shop_id','item_id']).cur_avg_prc.shift(1).fillna(sales_temp_new['cur_avg_prc'])
sales_temp_new1['last_cnt']=sales_temp_new.groupby(['shop_id','item_id']).cur_mon_cnt.shift(1).fillna(sales_temp_new['cur_mon_cnt'])
sales_temp_new1['last_mon']=sales_temp_new.groupby(['shop_id','item_id']).date_block_num.shift(1).fillna(sales_temp_new['date_block_num'])
# sales_temp_new1['last_year']=sales_temp_new.groupby(['shop_id','item_id']).year.shift(1).fillna(sales_temp_new['year'])
sales_temp_new1[['last_shop_id','last_item_id']]=sales_temp_new[['shop_id','item_id']]
df2=pd.concat([sales_temp_new, sales_temp_new1],axis=1)
# df2=df2[['shop_id_shift','item_id_shift','mon_shift','year_shift','last_prc','last_cnt','shop_id','item_id','date_block_num','year','cur_avg_prc','cur_mon_cnt']]
# df2=df2[['shop_id','item_id','date_block_num','year','cur_avg_prc','cur_mon_cnt','shop_id_shift','item_id_shift','mon_shift','year_shift','last_prc','last_cnt']]
df2[(df2['shop_id']==35)].head()
# sales_temp5[(sales_temp5['shop_id']==35)
#FOLLOWING DID NOT WORK as prc is considered as column name while shifting 

df3=df2.copy()
for i in range (1,7):
    prc='last_prc'+str(i-1)
    cnt='last_cnt'+str(i-1)
    mon='last_mon'+str(i-1)
    shop_id='last_shop_id'+str(i)
    item_id='last_item_id'+str(i)
    year='last_year'+str(i-1)
#     print (last_prc, last_cnt)
    df3['last_prc'+str(i)]=df3.groupby(['shop_id','item_id']).last_prc.shift(i).fillna(df3['last_prc'])
    df3['last_cnt'+str(i)]=df3.groupby(['shop_id','item_id']).last_cnt.shift(i).fillna(df3['last_cnt'])
    df3['last_mon'+str(i)]=df3.groupby(['shop_id','item_id']).last_mon.shift(i).fillna(df3['last_mon'])
    df3[[shop_id,item_id]]=df3[['shop_id','item_id']]
#     df3['last_year'+str(i)]=df3.groupby(['shop_id','item_id']).last_year.shift(i).fillna(df3['last_year'])
    
df3[(df3['shop_id']==35)].head(10)
# df3[(df3['shop_id']==35)].head(10)
df3=df3[df3.columns[::-1]]
df3[(df3['shop_id']==35)].head(10)

# block
df4=df3.copy()
idx=df4.index[(df4["year"]==df4["year"].max())  & (df4["date_block_num"]==33)]#& (sales_temp4["mon"]==10)
print ('test :-' ,df4.loc[idx].count()[0])#test  31531
test_new=df4.loc[idx].values[:,:-6]

df4= df4.drop(df4.index[idx])
print('train :-', df4.count()[0]) # train  1577593
train_new=df4.values[:,:-6]

# df5=df5.drop(["shop_id","item_id","date_block_num","year","cur_avg_prc"],axis=1)
print(df3[df3['shop_id']==35].head().values[1,:-6][:-1])
# print(df5[df5['shop_id_shift']==35].head().values[1,5:])
# sales_vals=sales_temp3.values
# train_new=df5.values[:,5:]
# test_new=df4.loc[idx].values[:,5:]
train_X, train_y = train_new[:,:-1 ], train_new[:, -1]
test_X, test_y = test_new[:, :-1], test_new[:, -1]
train_X = train_X.reshape((train_X.shape[0], 7, 5))
test_X = test_X.reshape((test_X.shape[0], 7, 5))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
import matplotlib.pyplot

model1 = Sequential()
model1.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(Dropout(0.4)) #loss: 0.9974 - val_loss: 2.8298
#model1.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(Dense(1))
opt=RMSprop(lr=0.002,decay=0.0002)
#opt=Adam()
model1.compile(loss='mean_squared_error', optimizer=opt)
# fit network
history = model1.fit(train_X, train_y, epochs=40, batch_size=256, validation_data=(test_X,test_y), verbose=2, shuffle=False)
# plot history

