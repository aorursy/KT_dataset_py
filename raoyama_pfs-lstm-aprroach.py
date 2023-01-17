import numpy as np

import pandas as pd 

import os
#データ読み込み

os.listdir('../input')

sales_data = pd.read_csv('../input/sales_train.csv')

item_cat = pd.read_csv('../input/item_categories.csv')

items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

test_data = pd.read_csv('../input/test.csv')

def basic_eda(df):

    print("----------TOP 5 RECORDS--------")

    print(df.head(5))

    print("----------INFO-----------------")

    print(df.info())

    print("----------Describe-------------")

    print(df.describe())

    print("----------Columns--------------")

    print(df.columns)

    print("----------Data Types-----------")

    print(df.dtypes)

    print("-------Missing Values----------")

    print(df.isnull().sum())

    print("-------NULL values-------------")

    print(df.isna().sum())

    print("-----Shape Of Data-------------")

    print(df.shape)

    

    
#Litle bit of exploration of data



print("=============================Sales Data=============================")

basic_eda(sales_data)

print("=============================Test data=============================")

basic_eda(test_data)

print("=============================Item Categories=============================")

basic_eda(item_cat)

print("=============================Items=============================")

basic_eda(items)

print("=============================Shops=============================")

basic_eda(shops)

print("=============================Sample Submission=============================")

basic_eda(sample_submission)



#'date'がobjectになっているので直す

sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')
#月次の売上データが欲しいので横にdate_block_num（月）を、縦にshop_id,item_id,item_cnt_day（日次の売上）

dataset = sales_data.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')

dataset.reset_index(inplace = True)

dataset.head()
#テストデータとmerge

dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')
# 欠損値を０に

dataset.fillna(0,inplace = True)

dataset.head()
# shop_idとitem_idはいらないので落とす

dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

dataset.head()
# 最終月を除いた月から、 

X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

# 最終月を求める

y_train = dataset.values[:,-1:]



# テストでは最初の月だけ省く

X_test = np.expand_dims(dataset.values[:,1:],axis = 2)



print(X_train.shape,y_train.shape,X_test.shape)

# ライブラリインポート

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
# モデルを定義

my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1)))

my_model.add(Dropout(0.4))

my_model.add(Dense(1))



my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

my_model.summary()

my_model.fit(X_train,y_train,batch_size = 2048,epochs = 5)
#提出ファイル作成

submission_pfs = my_model.predict(X_test)

submission_pfs = submission_pfs.clip(0,20)

submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})

submission.to_csv('sub_pfs.csv',index = False)
