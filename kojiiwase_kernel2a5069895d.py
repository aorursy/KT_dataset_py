%cd /content/drive/My Drive/Colab Notebooks/Kaggle/Predict Future Sales
import numpy as np

import pandas as pd

# pandasで表示が省略されるのを防ぐ

pd.set_option('display.max_rows',500)

pd.set_option('display.max_columns',100)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from itertools import product

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error 



#LightGBMライブラリ

import lightgbm as lgb
items = pd.read_csv('./input/items.csv')

shops = pd.read_csv('./input/shops.csv')

cats = pd.read_csv('./input/item_categories.csv')

train = pd.read_csv('./input/sales_train.csv')

# IDをindexにおく

test  = pd.read_csv('./input/test.csv').set_index('ID')
test
plt.figure(figsize=(10,4))

plt.xlim(-100,3000)

sns.boxplot(x=train['item_cnt_day'])



plt.figure(figsize=(10,4))

plt.xlim(-1000,train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)

# おおきすぎる外れ値は外してやる

train=train[train.item_price<100000]

train=train[train.item_cnt_day<1500]
# グラフからitem_priceが負のあたい,（NaN）があるとわかる

# NANは代表値で埋めてやる



# まずnanを確認

train[train.item_price<0]

# 代表値の取り方

# shop_id、item_id、date_block_num、が同じものの中からその平均値をとってやる

#月によって、その店のその商品の価格は変動しているから実際は日ごとに変動しているらしい

mean=train[(train.item_id==2973)&(train.shop_id==32)&(train.date_block_num==4)&(train.date_block_num==4)&(train.item_price>0)].item_price.mean()

train.loc[train.item_price<0,'item_price']=mean

# train[条件式,条件によってしていされたDFのカラムを指定]=X

# で、その条件式を満たすDFの指定したカラムにXが入る
train
train.iloc[484683]
# item_cnt_dayの負の値も見ていく

train[train.item_cnt_day<0]


train[train.item_cnt_day<0].item_cnt_day.value_counts()
# このマイナスの値は返品？という可能性を考えてとりあえずそのまま放置する

# 他の人のノートブックをみても放置している人が多かった
# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
# Сергиев ПосадはСергиевПосадをcity名にするといいらしいのでその処理

shops.loc[shops.shop_name=='Сергиев Посад ТЦ "7Я"','shop_name']='СергиевПосад ТЦ "7Я"'

shops['city']=shops['shop_name'].str.split(' ').map(lambda x:x[0])

# !ЯкутскとЯкутскはおなじものらしいのでマージしてやる

shops.loc[shops.city=='!Якутск','city']='Якутск'

# ラベルエンコーディングでcity_nameダミー化

shops['city_code']=LabelEncoder().fit_transform(shops['city'])

shops=shops[['shop_id','city_code']]
cats['split']=cats['item_category_name'].str.split('-')

cats['type']=cats['split'].map(lambda x:x[0].strip())

cats['type_code']=LabelEncoder().fit_transform(cats['type'])

# subtypeが存在しないばあいはtypeを入れる

cats['subtype']=cats['split'].map(lambda x:x[1].strip() if len(x)>1 else x[0].strip())

cats['subtype_code']=LabelEncoder().fit_transform(cats['subtype'])



cats=cats[['item_category_id','type_code','subtype_code']]



cats
items
items.drop(['item_name'],axis=1,inplace=True)
def train_test_only1(columns):

  print('test_only,test,train,train_only')

  print(len(list(set(test[columns]) - set(test[columns]).intersection(set(train[columns])))),len(list(set(test[columns]))),len(list(set(train[columns]))),len(list(set(train[columns]) - set(train[columns]).intersection(set(test[columns])))))
train_test_only1('item_id')
train_test_only1('shop_id')
matrix=[]

cols=['date_block_num','shop_id','item_id']

for i in range(34):

  sales=train[train.date_block_num==i]

  matrix.append(np.array(list(product([i],sales.shop_id.unique(),sales.item_id.unique())),dtype='int16'))
matrix
# 配列をたてに結合してDF化

matrix=pd.DataFrame(np.vstack(matrix),columns=cols)
# date_block_num,shop_id,item_idの優先度で昇順に並び替え

matrix.sort_values(cols,inplace=True)
# revenue（歳入）のカラムを作る

train['revenue']=train['item_price']*train['item_cnt_day']
group=train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':['sum']})

group.columns=['item_cnt_month']

group.reset_index(inplace=True)

group
matrix=pd.merge(matrix,group,on=cols,how='left')

matrix['item_cnt_month']=(matrix['item_cnt_month']

                          .fillna(0)

                          .clip(0,20))
matrix

test['date_block_num']=34

test
matrix=pd.concat([matrix,test],ignore_index=True,sort=False,keys=cols)

matrix
matrix.fillna(0,inplace=True)
matrix=pd.merge(matrix,shops,on=['shop_id'],how='left')

matrix=pd.merge(matrix,items,on=['item_id'],how='left')

matrix=pd.merge(matrix,cats,on=['item_category_id'],how='left')



matrix
# lag特徴量を作る関数の作成

def lag_feature(df,lags,col):

  tmp=df[['date_block_num','shop_id','item_id',col]]

  for i in lags:

    shifted=tmp.copy()

    shifted.columns=['date_block_num','shop_id','item_id',col+'_lag_'+str(i)]

    shifted['date_block_num']+=i

    df=pd.merge(df,shifted,on=['date_block_num','shop_id','item_id'],how='left')

  return df
# item_cnt_monthのラグを取る

matrix=lag_feature(matrix,[1,2,3,6,12],'item_cnt_month')

matrix
# 月ごとに一日何個売れたかの平均を取る

group=matrix.groupby(['date_block_num']).agg({'item_cnt_month':['mean']})

group.columns=['date_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num'],how='left')

matrix=lag_feature(matrix,[1,2],'date_avg_item_cnt')

matrix.drop(['date_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊item_idごとの平均を取る

group=matrix.groupby(['date_block_num','item_id']).agg({'item_cnt_month':['mean']})

group.columns=['date_item_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','item_id'],how='left')

matrix=lag_feature(matrix,[1,2,3,6,12],'date_item_avg_item_cnt')

matrix.drop(['date_item_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊shop_idごとの平均を取る

group=matrix.groupby(['date_block_num','shop_id']).agg({'item_cnt_month':['mean']})

group.columns=['date_shop_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','shop_id'],how='left')

matrix=lag_feature(matrix,[1,2,3,6,12],'date_shop_avg_item_cnt')

matrix.drop(['date_shop_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊item_category_idごとの平均を取る

group=matrix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':['mean']})

group.columns=['date_cat_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','item_category_id'],how='left')

matrix=lag_feature(matrix,[1,2],'date_cat_avg_item_cnt')

matrix.drop(['date_cat_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊shop_id*item_categoryごとの平均を取る

group=matrix.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month':['mean']})

group.columns=['date_shop_cat_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','shop_id','item_category_id'],how='left')

matrix=lag_feature(matrix,[1,2],'date_shop_cat_avg_item_cnt')

matrix.drop(['date_shop_cat_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊shop_id*type_codeごとの平均を取る

group=matrix.groupby(['date_block_num','shop_id','type_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_shop_type_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','shop_id','type_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_shop_type_avg_item_cnt')

matrix.drop(['date_shop_type_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊shop_id*subtype_codeごとの平均を取る

group=matrix.groupby(['date_block_num','shop_id','subtype_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_shop_subtype_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','shop_id','subtype_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_shop_subtype_avg_item_cnt')

matrix.drop(['date_shop_subtype_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊city_codeごとの平均を取る

group=matrix.groupby(['date_block_num','city_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_city_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','city_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_city_avg_item_cnt')

matrix.drop(['date_city_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊type_codeごとの平均を取る

group=matrix.groupby(['date_block_num','type_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_type_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','type_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_type_avg_item_cnt')

matrix.drop(['date_type_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊subtype_codeごとの平均を取る

group=matrix.groupby(['date_block_num','subtype_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_subtype_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','subtype_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_subtype_avg_item_cnt')

matrix.drop(['date_subtype_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊city_code*subtype_codeごとの平均を取る

group=matrix.groupby(['date_block_num','city_code','subtype_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_city_subtype_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','city_code','subtype_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_city_subtype_avg_item_cnt')

matrix.drop(['date_city_subtype_avg_item_cnt'],axis=1,inplace=True)
# date_block_num＊city_code*subtype_codeごとの平均を取る

group=matrix.groupby(['date_block_num','city_code','type_code']).agg({'item_cnt_month':['mean']})

group.columns=['date_city_type_avg_item_cnt']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','city_code','type_code'],how='left')

matrix=lag_feature(matrix,[1,2],'date_city_type_avg_item_cnt')

matrix.drop(['date_city_type_avg_item_cnt'],axis=1,inplace=True)
matrix
# item_idごとにitem_priceの平均を取る

group=train.groupby(['item_id']).agg({'item_price':['mean']})

group.columns=['item_avg_item_price']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['item_id'],how='left')

# date_block_num*item_idごとのitem_priceの平均をもとめる

group=train.groupby(['date_block_num','item_id']).agg({'item_price':['mean']})

group.columns=['date_item_avg_item_price']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','item_id'],how='left')

# trainデータ全体から、item_idごとにitem_priceの平均を取ったものと、月ごとitem_idごとにitem_priceの平均を取ったものとの誤差率をだす

matrix['dif_date_item_price']=  (matrix['date_item_avg_item_price'] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']



# ラグを取る

matrix=lag_feature(matrix,[1,2],'dif_date_item_price')
# date_block_num*shop_idごとのrevenueの平均をもとめる

group=train.groupby(['date_block_num','shop_id']).agg({'revenue':['mean']})

group.columns=['date_shop_avg_revenue']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['date_block_num','shop_id'],how='left')

# shop_idごとのrevenueの平均をもとめる

group=train.groupby(['shop_id']).agg({'revenue':['mean']})

group.columns=['shop_avg_revenue']

group.reset_index(inplace=True)

matrix=pd.merge(matrix,group,on=['shop_id'],how='left')



# trainデータ全体から、item_idごとにrevenueの平均を取ったものと、月ごとitem_idごとにrevenueの平均を取ったものとの誤差率をだす

matrix['dif_date_revenue']=  (matrix['date_shop_avg_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
# ラグを取る

matrix=lag_feature(matrix,[1,2],'dif_date_item_price')
# 月を記述

matrix['month']=matrix['date_block_num']%12

# 月の日数を示すカラムを追加

days=pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

matrix['days']=matrix['month'].map(days)
# 12か月まえまでのラグを取っているので、date_block_numの0~11までを消してやる

matrix=matrix[matrix['date_block_num']>11]

matrix
matrix.isnull().any()
matrix.fillna(0,inplace=True)
matrix.isnull().any()
matrix.info()
# pklデータに落とし込む

matrix.to_pickle('data.pkl')
# 再読み込み

data = pd.read_pickle('data.pkl')
data.columns
data=data[[

           'date_block_num', 

           'shop_id', 

           'item_id', 

           'item_cnt_month', 

           'city_code',

           'item_category_id', 

           'type_code',  

           'subtype_code',  

           'item_cnt_month_lag_1',

           'item_cnt_month_lag_2',  

           'item_cnt_month_lag_3', 

           'item_cnt_month_lag_6',

           'item_cnt_month_lag_12', 

           'date_avg_item_cnt_lag_1',

           'date_avg_item_cnt_lag_2', 

           'date_item_avg_item_cnt_lag_1',

           'date_item_avg_item_cnt_lag_2',  

           'date_item_avg_item_cnt_lag_3',

           'date_item_avg_item_cnt_lag_6', 

           'date_item_avg_item_cnt_lag_12',

           'date_shop_avg_item_cnt_lag_1', 

           'date_shop_avg_item_cnt_lag_2',

           'date_shop_avg_item_cnt_lag_3', 

           'date_shop_avg_item_cnt_lag_6',

           'date_shop_avg_item_cnt_lag_12', 

           'date_cat_avg_item_cnt_lag_1',

           'date_cat_avg_item_cnt_lag_2', 

           'date_shop_cat_avg_item_cnt_lag_1',

           'date_shop_cat_avg_item_cnt_lag_2', 

           'date_shop_type_avg_item_cnt_lag_1',

           'date_shop_type_avg_item_cnt_lag_2',

           'date_shop_subtype_avg_item_cnt_lag_1',

           'date_shop_subtype_avg_item_cnt_lag_2', 

           'date_city_avg_item_cnt_lag_1',

           'date_city_avg_item_cnt_lag_2', 

           'date_type_avg_item_cnt_lag_1',

           'date_type_avg_item_cnt_lag_2', 

           'date_subtype_avg_item_cnt_lag_1',

           'date_subtype_avg_item_cnt_lag_2',

           'date_city_subtype_avg_item_cnt_lag_1',

           'date_city_subtype_avg_item_cnt_lag_2',

           'date_city_type_avg_item_cnt_lag_1',

           'date_city_type_avg_item_cnt_lag_2',   

           'item_avg_item_price',

          #  'date_item_avg_item_price', 

          #  'dif_date_item_price',

           'dif_date_item_price_lag_1_x', 

           'dif_date_item_price_lag_2_x',

          #  'date_shop_avg_revenue', 

           'shop_avg_revenue',

          #  'dif_date_revenue',

          #  'dif_date_item_price_lag_1_y', 

          #  'dif_date_item_price_lag_2_y', 

           'month',

           'days'

           



















]]
data
X_train=data[data.date_block_num<33].drop(['item_cnt_month'],axis=1)

y_train=data[data.date_block_num<33]['item_cnt_month']

X_valid=data[data.date_block_num==33].drop(['item_cnt_month'],axis=1)

y_valid=data[data.date_block_num==33]['item_cnt_month']

X_test=data[data.date_block_num==34].drop(['item_cnt_month'],axis=1)



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid)
# パラメータサーチ

RMSE_list = []

count = []

for i in range(6, 13):

    params = {'boosting_type': 'gbdt',

          'objective': 'regression',

          'metric': 'rmse',

          'max_depth' : i}

    

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=lgb_eval,

                early_stopping_rounds=100,

                verbose_eval=50)

    

    predicted = gbm.predict(X_valid)

    pred_df = pd.concat([y_valid.reset_index(drop=True), pd.Series(predicted)], axis=1)

    pred_df.columns = ['true', 'pred']

    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))

    RMSE_list.append(RMSE)

    count.append(i)
params = {'boosting_type': 'gbdt',

         'objective': 'regression',

          'metric': 'rmse',

          'max_depth' : 8}

    



gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=lgb_eval,

                early_stopping_rounds=100,

                verbose_eval=50)
# model=xgb.XGBRegressor(

#     min_child_weight=300, 

#     colsample_bytree=0.8, 

#     subsample=0.8, 

#     eta=0.3,    

#     seed=42,

#     # ここに最適ハイパーパラメータを記入

#     max_depth=8,

#     n_estimators=1000

# )
# model.fit(

#     X_train,

#     y_train,

#     eval_metric='rmse',

#     eval_set=[(X_train,y_train),(X_valid,y_valid)],

#     early_stopping_rounds=10

# )
# y_pred=model.predict(X_test).clip(0,20)

# submission=pd.DataFrame([

#     'ID':test.index,

#     'item_cnt_month':y_pred

                         



# ])
%tensorflow_version 1.x
import tensorflow as tf

import keras as keras

from keras.models import Sequential

from keras.layers import  LSTM,Dense,Dropout,RNN

from sklearn.metrics import mean_squared_error





model=Sequential()



model = Sequential()

model.add(LSTM(units=64, input_shape=(1, 48),dropout=0.1,recurrent_dropout=0.1))

model.add(Dense(1))



model.compile(loss='mse',

              optimizer='adam',

              metrics=['mean_squared_error'])

model.summary()



X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))





X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))



X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))
eval_set=(X_valid,y_valid)
es_cb=keras.callbacks.EarlyStopping( patience=2, verbose=0)
history=model.fit(X_train,y_train,batch_size=2048,epochs=200,validation_data=eval_set,callbacks=[es_cb])
model.save('model.h5', include_optimizer=False)
# y_predを0から20の範囲に収めて、四捨五入

y_pred=model.predict(X_test).clip(0,20).round()
y_pred.shape
y_pred=y_pred.reshape(y_pred.shape[0])
y_pred
submissionLSTM = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": y_pred

})
submissionLSTM[submissionLSTM.item_cnt_month>=1]
submissionLSTM
submissionLSTM
submissionLSTM.to_csv('LSTM.csv',index=False)