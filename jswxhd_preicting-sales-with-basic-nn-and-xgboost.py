import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import itertools
import gc
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
tf.__version__
ic = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
item = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shop = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
test.head() 
#test.head()
#len(shop['shop_id'].value_counts())
#len(i['item_id'].value_counts())
#len(ic['item_category_id'].value_counts())
average_price = pd.DataFrame(train['item_price'].groupby(train['item_id']).mean())
print(average_price.describe())
plt.figure(figsize=(20,8))
plt.hist(average_price['item_price'],bins = 50,range=(0,2000))
plt.xlabel('Item Prices')
plt.ylabel('Frequency')
plt.title('Distribution of Item Prices')
plt.show()
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.boxplot(train['item_price'])
plt.title('Item Price')
plt.subplot(122)
plt.boxplot(train['item_cnt_day'])
plt.title('Item count Day')
plt.tight_layout(pad=3)
plt.show()
train = train[(train['item_price']<10000) & (train['item_cnt_day']<1001)]
negp = train[train['item_price']<=0]
print(negp)
median = train[(train.shop_id==32)&(train.item_id==2973)&
               (train.date_block_num==4)&(train.item_price>0)]['item_price'].median()
train.loc[train.item_price<0,'item_price'] = median
shop['shop_name'].groupby(shop.shop_id).value_counts()
train.loc[train.shop_id == 0,'shop_id']=57
test.loc[test.shop_id == 0,'shop_id']=57
train.loc[train.shop_id == 1,'shop_id']=58
test.loc[test.shop_id == 1,'shop_id']=58
train.loc[train.shop_id == 10,'shop_id']=11
test.loc[test.shop_id == 10,'shop_id']=11
ic['split'] = ic['item_category_name'].str.split('-')
ic['type'] = ic['split'].map(lambda x:x[0].strip())
ic['type_code'] = LabelEncoder().fit_transform(ic['type'])
ic['subtype'] = ic['split'].map(lambda x: x[1].strip() if len(x)>1 else x[0].strip())
ic['subtype_code'] = LabelEncoder().fit_transform(ic['subtype'])
ic = ic[['item_category_id','type_code','subtype_code']]
ic.head()
item.drop(['item_name'],axis=1,inplace=True)
item.head()
train[(train['shop_id'] == 5) & (train['item_id']==5320)]#['item_cnt_day'].groupby(train['date_block_num']).sum()
train_match = []
for i in range(34):
    sales = train[train.date_block_num==i]
    train_match.append(np.array(list(itertools.product([i],sales.shop_id.unique(),
                                        sales.item_id.unique())),dtype='int16'))

train_match[:5]
#np.shape(train_match)
train_match = pd.DataFrame(np.vstack(train_match),columns =['date_block_num','shop_id','item_id'])
train_match.sort_values(by = ['date_block_num','shop_id','item_id'],inplace=True)
train_match.head()
grouped = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':['sum']})
grouped.columns = ['item_cnt_month']
grouped.reset_index(inplace = True)
train_match = pd.merge(train_match,grouped,on=['date_block_num','shop_id','item_id'],how='left')
train_match.head()
train_match['item_cnt_month'] = (train_match['item_cnt_month']
                                 .fillna(0).clip(0,20).astype(np.float16))
train_match.head()
test['date_block_num'] = 34
train_match = pd.concat([train_match,test],ignore_index=True,
                        sort = False,keys = ['date_block_num','shop_id','item_id'])
train_match.fillna(0,inplace=True)
train_match.tail()
train_match = pd.merge(train_match, item, on=['item_id'], how = 'left')
train_match = pd.merge(train_match, ic, on=['item_category_id'], how = 'left')
train_match.head()
def create_lag_feature(df,lags,col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for lag in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id',col+'_lag_'+str(lag)]
        ##col value is the timestamp 'lag' (months/date_block_num) before 
        shifted['date_block_num'] += lag
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'],how = 'left')
    
    return df
train_match = create_lag_feature(train_match,[1,2,3,6,12],'item_cnt_month')
train_match.head()
grouped2 = train_match.groupby(['date_block_num']).agg({'item_cnt_month':['mean']})
grouped2.columns = ['avg_item_cnt_month']
grouped2.reset_index(inplace = True)
grouped2.head()
train_match = pd.merge(train_match, grouped2, on=['date_block_num'], how='left')
train_match['avg_item_cnt_month'] = train_match['avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1],'avg_item_cnt_month')
train_match.drop(['avg_item_cnt_month'],axis =1, inplace = True)
train_match.sample(5)
grouped3 = train_match.groupby(['date_block_num','item_id']).agg({'item_cnt_month':['mean']})
grouped3.columns = ['item_avg_item_cnt_month']
grouped3.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped3, on=['date_block_num','item_id'], how ='left')
train_match['item_avg_item_cnt_month'] = train_match['item_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1,2,3,6,12], 'item_avg_item_cnt_month')
train_match.drop(['item_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped4 = train_match.groupby(['date_block_num','shop_id']).agg({'item_cnt_month':['mean']})
grouped4.columns = ['shop_avg_item_cnt_month']
grouped4.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped4, on=['date_block_num','shop_id'], how ='left')
train_match['shop_avg_item_cnt_month'] = train_match['shop_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1,2,3,6,12], 'shop_avg_item_cnt_month')
train_match.drop(['shop_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped5 = train_match.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':['mean']})
grouped5.columns = ['ic_avg_item_cnt_month']
grouped5.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped5, on=['date_block_num','item_category_id'], how ='left')
train_match['ic_avg_item_cnt_month'] = train_match['ic_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1], 'ic_avg_item_cnt_month')
train_match.drop(['ic_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped6 = train_match.groupby(['date_block_num','type_code']).agg({'item_cnt_month':['mean']})
grouped6.columns = ['type_avg_item_cnt_month']
grouped6.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped6, on=['date_block_num','type_code'], how ='left')
train_match['type_avg_item_cnt_month'] = train_match['type_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1], 'type_avg_item_cnt_month')
train_match.drop(['type_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped7 = train_match.groupby(['date_block_num','subtype_code']).agg({'item_cnt_month':['mean']})
grouped7.columns = ['subtype_avg_item_cnt_month']
grouped7.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped7, on=['date_block_num','subtype_code'], how ='left')
train_match['subtype_avg_item_cnt_month'] = train_match['subtype_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1], 'subtype_avg_item_cnt_month')
train_match.drop(['subtype_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped8 = train_match.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month':['mean']})
grouped8.columns = ['shop_ic_avg_item_cnt_month']
grouped8.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped8, on=['date_block_num','shop_id','item_category_id'], how ='left')
train_match['shop_ic_avg_item_cnt_month'] = train_match['shop_ic_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1], 'shop_ic_avg_item_cnt_month')
train_match.drop(['shop_ic_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped9 = train_match.groupby(['date_block_num','shop_id','type_code']).agg({'item_cnt_month':['mean']})
grouped9.columns = ['shop_type_avg_item_cnt_month']
grouped9.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped9, on=['date_block_num','shop_id','type_code'], how ='left')
train_match['shop_type_avg_item_cnt_month'] = train_match['shop_type_avg_item_cnt_month'].astype('float16')
train_match = create_lag_feature(train_match, [1], 'shop_type_avg_item_cnt_month')
train_match.drop(['shop_type_avg_item_cnt_month'], axis =1, inplace =True)
train_match.sample(5)
grouped10 = train.groupby(['item_id']).agg({'item_price':['mean']})
grouped10.columns = ['item_avg_item_price']
grouped10.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped10, on=['item_id'], how='left')
train_match['item_avg_item_price'] = train_match['item_avg_item_price'].astype('float16')

grouped11 = train.groupby(['date_block_num','item_id']).agg({'item_price':['mean']})
grouped11.columns = ['item_avg_item_price_month']
grouped11.reset_index(inplace = True)
train_match = pd.merge(train_match, grouped11, on=['date_block_num','item_id'], how='left')
train_match['item_avg_item_price_month'] = train_match['item_avg_item_price_month'].astype('float16')

all_lags = [1,2,3,4,5,6]
train_match = create_lag_feature(train_match,all_lags,'item_avg_item_price_month')
for l in all_lags:
    train_match['delta_price_lag_'+str(l)]= (train_match['item_avg_item_price_month_lag_'+str(l)] - train_match['item_avg_item_price'])/train_match['item_avg_item_price']

def valid_trend(row):
    for l in all_lags:
        if row['delta_price_lag_'+str(l)]:
            return row['delta_price_lag_'+str(l)]
    return 0

train_match['delta_price_lag'] = train_match.apply(valid_trend, axis=1)
train_match['delta_price_lag'] = train_match['delta_price_lag'].astype('float16')
train_match['delta_price_lag'].fillna(0, inplace =True)

to_drop = ['item_avg_item_price','item_avg_item_price_month']
for l in all_lags:
    to_drop += ['item_avg_item_price_month_lag_'+str(l)]
    to_drop += ['delta_price_lag_'+str(l)]
    
train_match.drop(to_drop, axis=1, inplace=True)
train_match.sample(5)
train_match['month'] = (train_match['date_block_num']%12)+1
train_match.sample(5)
sale_record = {}
train_match['last_item_sale'] = -1
train_match['last_item_sale'] = train_match['last_item_sale'].astype('int8')
for idx,row in train_match.iterrows():
    key = row.item_id
    if key not in sale_record.keys():
        if row.item_cnt_month != 0:
            sale_record[key] = row.date_block_num
            
    else:
        last_date_block_num = sale_record[key]
        if row.date_block_num > last_date_block_num:
            train_match.at[idx,'last_item_sale'] = row.date_block_num - last_date_block_num
            sale_record[key] = row.date_block_num
            
train_match.sample(5)
train_match = train_match[train_match['date_block_num']>11]
def fillna_lag(df):
    for col in df.columns:
        if '_lag_' in col and df[col].isnull().any():
            if 'item_cnt' in col:
                df[col].fillna(0, inplace = True)
                
    return df

train_match = fillna_lag(train_match)
train_match.info()
train_match.to_pickle('training.pkl')
del train_match
del sale_record  
del item
del shop
del ic
del train
del grouped
del grouped2
del grouped3
del grouped4
del grouped5
del grouped6
del grouped7
del grouped8
del grouped9
del grouped10
del grouped11

gc.collect()
training = pd.read_pickle('../input/training-data-for-item-sale-predict/training.pkl')
train_x = training[training.date_block_num<33].drop(['item_cnt_month'],axis=1)
train_y = training[training.date_block_num<33]['item_cnt_month']
valid_x = training[training.date_block_num==33].drop(['item_cnt_month'],axis=1)
valid_y = training[training.date_block_num==33]['item_cnt_month']
test_x = training[training.date_block_num==34].drop(['item_cnt_month'],axis=1)
max_depth_range = {'max_depth':range(6,10,1)}
cv1 = GridSearchCV(estimator = XGBRegressor(n_estimators = 1000,  
                                            eta = 0.3,
                                            subsample=0.8,
                                            colsample_bytree=0.8,
                                            seed = 0),
                  param_grid = max_depth_range,
                  scoring = 'neg_root_mean_squared_error')

cv1.fit(train_x,train_y)
cv1.cv_results_, cv1.best_score_, cv1.best_params_
xgb_model = XGBRegressor(n_estimators = 1500, 
                         max_depth = 8, 
                         eta = 0.1,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         min_child_weight = 3,
                         seed = 0)
xgb_param = xgb_model.get_xgb_params()
xg_train  = xgb.DMatrix(train_x, train_y)
cv_res = xgb.cv(xgb_param, xg_train, num_boost_round =1000, nfold =3,
               metrics ='rmse', early_stopping_rounds =10, stratified =True, 
                seed=0)
cv_res
xgb_model.fit(train_x,train_y,
             eval_metric = 'rmse',
             eval_set = [(train_x, train_y), (valid_x, valid_y)],
             early_stopping_rounds = 10)
plot_importance(xgb_model,height =0.1,max_num_features = 20)
result = xgb_model.predict(test_x).clip(0,20)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
submission = pd.DataFrame({'ID':test.index, 'item_cnt_month':result})
submission.to_csv('xgb_submission.csv',index =False)
def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def MLP_model():
    model = Sequential()
    model.add(Dense(256, activation ='relu',input_dim =30))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation ='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1))
    adam = Adam(lr =5e-5)
    model.compile(loss ='mse', optimizer =adam, metrics =[rmse])
    
    return model
base_model = MLP_model()
base_model.fit(train_x, train_y, batch_size =32, epochs =15, validation_data =(valid_x, valid_y))
base_model.summary()
base_model.save('baseline_model.h5')
result = base_model.predict(test_x).clip(0,20)
result = [each for r in result for each in r]
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
submission = pd.DataFrame({'ID':test.index, 'item_cnt_month':result})
submission.to_csv('baseline_submission.csv',index =False)
valid_result = base_model.predict(valid_x).clip(0,20)
valid_result = [each for r in valid_result for each in r]
plt.figure(figsize=(10,8))
plt.plot(range(100),valid_result[:100], color='blue', label= 'valid_pred')
plt.plot(range(100),valid_y[:100], color = 'green',label='valid')
plt.legend()
plt.show()