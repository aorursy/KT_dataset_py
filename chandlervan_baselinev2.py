import warnings
warnings.filterwarnings('ignore')
import gc
import os
import sys
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import Parallel,delayed
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
def datetime_transform(df):
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = pd.to_datetime(df['first_active_month']).dt.year
    df['month'] = pd.to_datetime(df['first_active_month']).dt.month
    return df


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=2)(delayed(func)(group)
                                for name, group in dfGrouped)
    return retLst


def get_stat_feature(df):
    city_cnt = df['city_id'].nunique()
    mer_cnt = df['merchant_id'].nunique()
    state_cnt = df['state_id'].nunique()
    installments = len(df['installments'] > 0)
    return [df['card_id'].values[0], city_cnt, mer_cnt, state_cnt, installments]
%%time
train = pd.read_csv('../input/train.csv',)
test = pd.read_csv('../input/test.csv')
ht = pd.read_csv('../input/historical_transactions.csv')
nmt = pd.read_csv('../input/new_merchant_transactions.csv')
#日期变换
train = datetime_transform(train)
test = datetime_transform(test)
ht['purchase_date'] = pd.to_datetime(ht['purchase_date'])
nmt['purchase_date'] = pd.to_datetime(nmt['purchase_date'])
ht['purchase_year'] = ht['purchase_date'].dt.year
nmt['purchase_year'] = nmt['purchase_date'].dt.year
ht['purchase_month'] = ht['purchase_date'].dt.month
nmt['purchase_month'] = nmt['purchase_date'].dt.month
#缺失值填充
ht['category_2'] = ht['category_2'].fillna(0)
ht['category_3'] = ht['category_3'].fillna('E')
nmt['category_2'] = nmt['category_2'].fillna(0)
nmt['category_3'] = nmt['category_3'].fillna('E')
ht['category_2'] = ht['category_2'].map(lambda x:str(int(x)))
nmt['category_2'] = nmt['category_2'].map(lambda x:str(int(x)))
#数据类型转换
ht['merchant_category_id'] = ht['merchant_category_id'].map(lambda x:str(x))
ht['merchant_id'] = ht['merchant_id'].map(lambda x:str(x))
nmt['merchant_category_id'] = nmt['merchant_category_id'].map(lambda x:str(x))
nmt['merchant_id'] = nmt['merchant_id'].map(lambda x:str(x))
#部分数据转换，节省空间
lb = LabelEncoder()
ht['authorized_flag'] = lb.fit_transform(ht['authorized_flag'])
lb = LabelEncoder()
ht['category_1'] = lb.fit_transform(ht['category_1'])
ht['category_2'] = ht['category_2'].astype(np.int16)
lb = LabelEncoder()
ht['category_3'] = lb.fit_transform(ht['category_3'])
ht['purchase_amount'] = ht['purchase_amount'].astype(np.float32)
ht['month_lag'] = ht['month_lag'].astype(np.float32)
ht['installments'] = ht['installments'].astype(np.int16)
ht['city_id'] = ht['city_id'].astype(np.int16)
lb = LabelEncoder()
ht['merchant_category_id'] = lb.fit_transform(ht['merchant_category_id'])
ht['state_id'] = ht['state_id'].astype(np.int16)
ht['subsector_id'] = ht['subsector_id'].astype(np.int16)
ht['purchase_year'] = ht['purchase_year'].astype(np.int16)
ht['purchase_month'] = ht['purchase_month'].astype(np.int16)
v = pd.pivot_table(ht,values=['purchase_year'],index='card_id',columns=['state_id'],aggfunc=len,fill_value=0)
v.columns = ['state_id_%s'%s for s in range(v.shape[1])]
v = v.reset_index()
train = pd.merge(train,v,on='card_id',how='left')
test = pd.merge(test,v,on='card_id',how='left')
#激活日期距离今天的时间
train['active_till_now'] = train['first_active_month'].map(lambda x:(datetime.datetime.today() - x).days)
test['active_till_now'] = test['first_active_month'].map(lambda x:(datetime.datetime.today() - x).days)
#统计历史上和在新商品上买过多少次东西
ht_cnt = ht.groupby('card_id')['city_id'].count().reset_index()
nmt_cnt = nmt.groupby('card_id')['city_id'].count().reset_index()
ht_cnt.columns = ['card_id','ht_cnt']
nmt_cnt.columns = ['card_id','nmt_cnt']
#merge the result
train = pd.merge(train,ht_cnt,on='card_id',how='left')
test = pd.merge(test,ht_cnt,on='card_id',how='left')
train = pd.merge(train,nmt_cnt,on='card_id',how='left')
test = pd.merge(test,nmt_cnt,on='card_id',how='left')
ht_amount = ht.groupby('card_id').agg({'purchase_amount':[np.max,np.mean,np.std]})
ht_amount.columns = ['amount_max','amount_mean','amount_std']
ht_amount = ht_amount.reset_index()
#计算历史上的消费状况
train = pd.merge(train,ht_amount,on='card_id',how='left')
test = pd.merge(test,ht_amount,on='card_id',how='left')
#计算历史上month_lag的情况
ht_month_lag = ht.groupby('card_id').agg({'month_lag':[np.min,np.mean,np.std]})
ht_month_lag.columns = ['lag_min','lag_mean','lag_std']
ht_month_lag = ht_month_lag.reset_index()
train = pd.merge(train,ht_month_lag,on='card_id',how='left')
test = pd.merge(test,ht_month_lag,on='card_id',how='left')
#计算历史上installments的情况
ht_installment = ht.groupby('card_id').agg({'installments':[np.max,np.min,np.mean,np.std]})
ht_installment.columns = ['ins_max','ins_min','ins_mean','ins_std']
ht_installment = ht_installment.reset_index()
train = pd.merge(train,ht_installment,on='card_id',how='left')
test = pd.merge(test,ht_installment,on='card_id',how='left')
#计算新商店的消费状况
nmt_amount = nmt.groupby('card_id').agg({'purchase_amount':[np.max,np.mean,np.std]})
nmt_amount.columns = ['nmt_amount_max','nmt_amount_mean','nmt_amount_std']
nmt_amount = nmt_amount.reset_index()
train = pd.merge(train,nmt_amount,on='card_id',how='left')
test = pd.merge(test,nmt_amount,on='card_id',how='left')
#计算新商店上month_lag的情况
nmt_month_lag = nmt.groupby('card_id').agg({'month_lag':[np.max,np.mean,np.std]})
nmt_month_lag.columns = ['nmt_lag_min','nmt_lag_mean','nmt_lag_std']
nmt_month_lag = nmt_month_lag.reset_index()
train = pd.merge(train,nmt_month_lag,on='card_id',how='left')
test = pd.merge(test,nmt_month_lag,on='card_id',how='left')
#计算新商店上installments的情况
nmt_installment = nmt.groupby('card_id').agg({'installments':[np.max,np.mean,np.std]})
nmt_installment.columns = ['nmt_ins_min','nmt_ins_mean','nmt_ins_std']
nmt_installment = nmt_installment.reset_index()
train = pd.merge(train,nmt_installment,on='card_id',how='left')
test = pd.merge(test,nmt_installment,on='card_id',how='left')
%%time
ht_res = applyParallel(ht.groupby('card_id'),get_stat_feature)
nmt_res = applyParallel(nmt.groupby('card_id'),get_stat_feature)
#merge the result
stat_df = pd.DataFrame(ht_res,columns=['card_id','city_cnt','mer_cnt','state_cnt','installments'])
nmt_stat_df = pd.DataFrame(nmt_res,columns=['card_id','city_cnt_nmt','mer_cnt_nmt','state_cnt_nmt','installments_num'])
train = pd.merge(train,stat_df,on='card_id',how='left')
test = pd.merge(test,stat_df,on='card_id',how='left')
train = pd.merge(train,nmt_stat_df,on='card_id',how='left')
test = pd.merge(test,nmt_stat_df,on='card_id',how='left')
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
#handler the numeric feature
drop_columns = ['first_active_month','card_id','feature_1','feature_2','feature_3','target','year','month']
num_feature = [v for v in train.columns if v not in drop_columns]
train_num_feature = train[num_feature].values
test_num_feature = test[num_feature].values
ss = StandardScaler()
train_num_feature = ss.fit_transform(train_num_feature)
test_num_feature = ss.transform(test_num_feature)
#stack the feature
train_feature = train_num_feature
test_feature = test_num_feature
#不用stacking
res_lgb = []
feature_imp = []
for tr,va in KFold(n_splits=10,random_state=2018).split(train_feature,train['target'].values):
    lgbmr = LGBMRegressor(num_leaves=16,n_estimators=100,colsample_bytree=0.7,subsample=0.7,)
    lgbmr.fit(train_feature[tr],train['target'].values[tr],
              eval_set=[(train_feature[tr],train['target'].values[tr]),(train_feature[va],train['target'].values[va])],
              eval_metric='rmse',
              verbose=50)
    feature_imp.append(lgbmr.feature_importances_)
    res_lgb.append(lgbmr.predict(test_feature))
f = np.mean(feature_imp,axis=0)
avg_res = np.mean(res_lgb,axis=0)
test['target'] = avg_res
test[['card_id','target']].to_csv('predict1216.csv',index=False)
