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
%%time
train = pd.read_csv('../input/train.csv',)
test = pd.read_csv('../input/test.csv')
ht = pd.read_csv('../input/historical_transactions.csv')
nmt = pd.read_csv('../input/new_merchant_transactions.csv')
#日期变换
train['first_active_month'] = pd.to_datetime(train['first_active_month'])
test['first_active_month'] = pd.to_datetime(test['first_active_month'])
ht['purchase_date'] = pd.to_datetime(ht['purchase_date'])
ht['purchase_year'] = ht['purchase_date'].dt.year
ht['purchase_year'] = ht['purchase_year'].astype(np.int32)
ht['purchase_hour'] = ht['purchase_date'].dt.hour
ht['purchase_month'] = ht['purchase_date'].dt.month
ht['purchase_month'] = ht['purchase_month'].astype(np.int16)
nmt['purchase_date'] = pd.to_datetime(nmt['purchase_date'])
nmt['purchase_year'] = nmt['purchase_date'].dt.year
nmt['purchase_month'] = nmt['purchase_date'].dt.month
nmt['purchase_hour'] = nmt['purchase_date'].dt.hour
#缺失值填充
ht['category_2'] = ht['category_2'].fillna(0)
ht['category_3'] = ht['category_3'].fillna('E')
nmt['category_2'] = nmt['category_2'].fillna(0)
nmt['category_3'] = nmt['category_3'].fillna('E')
ht['category_2'] = ht['category_2'].astype(np.int16)
nmt['category_2'] = nmt['category_2'].astype(np.int16)
train['feature_1'] = train['feature_1'].astype(str)
train['feature_2'] = train['feature_2'].astype(str)
train['feature_3'] = train['feature_3'].astype(str)
test['feature_1'] = test['feature_1'].astype(str)
test['feature_2'] = test['feature_2'].astype(str)
test['feature_3'] = test['feature_3'].astype(str)
#激活日期距离今天的时间
train['active_till_now'] = train['first_active_month'].map(lambda x:(datetime.datetime.today() - x).days)
test['active_till_now'] = test['first_active_month'].map(lambda x:(datetime.datetime.today() - x).days)
#统计在24小时内的购买记录次数
hour_purchase_cnt = pd.pivot_table(ht,values='month_lag',index='card_id',columns=['purchase_hour'],aggfunc=len,fill_value=0)
hour_purchase_cnt.columns = ['purchase_hour_%s'%str(i) for i in range(24)]
hour_purchase_cnt.reset_index(inplace=True)
train = pd.merge(train,hour_purchase_cnt,on='card_id',how='left')
test = pd.merge(test,hour_purchase_cnt,on='card_id',how='left')
del hour_purchase_cnt
gc.collect()
#统计在24小时内的购买记录次数
hour_sub_id = pd.pivot_table(ht,values='month_lag',index='card_id',columns=['subsector_id'],aggfunc=len,fill_value=0)
hour_sub_id.columns = ['sub_id_%s'%str(i) for i in range(ht['subsector_id'].nunique())]
hour_sub_id.reset_index(inplace=True)
train = pd.merge(train,hour_sub_id,on='card_id',how='left')
test = pd.merge(test,hour_sub_id,on='card_id',how='left')
del hour_sub_id
gc.collect()
#统计在历史上每个月的购买次数
ym_purchase_cnt = pd.pivot_table(ht,values='month_lag',index='card_id',columns=['purchase_year','purchase_month'],aggfunc=len,fill_value=0)
ym_purchase_cnt.columns = ['purchase_ym_%s'%str(i) for i in range(ym_purchase_cnt.shape[1])]
ym_purchase_cnt.reset_index(inplace=True)
train = pd.merge(train,ym_purchase_cnt,on='card_id',how='left')
test = pd.merge(test,ym_purchase_cnt,on='card_id',how='left')
del ym_purchase_cnt
gc.collect()
#统计在24小时内的平均购买金额
hour_purchase_amount = pd.pivot_table(ht,values='purchase_amount',index='card_id',columns=['purchase_hour'],aggfunc=np.mean,fill_value=0)
hour_purchase_amount.columns = ['purchase_amount_%s'%str(i) for i in range(24)]
hour_purchase_amount.reset_index(inplace=True)
train = pd.merge(train,hour_purchase_amount,on='card_id',how='left')
test = pd.merge(test,hour_purchase_amount,on='card_id',how='left')
del hour_purchase_amount
gc.collect()
#统计在24小时内的平均month_lag
hour_month_lag = pd.pivot_table(ht,values='month_lag',index='card_id',columns=['purchase_hour'],aggfunc=np.sum,fill_value=0)
hour_month_lag.columns = ['month_lag_%s'%str(i) for i in range(24)]
hour_month_lag.reset_index(inplace=True)
train = pd.merge(train,hour_month_lag,on='card_id',how='left')
test = pd.merge(test,hour_month_lag,on='card_id',how='left')
del hour_month_lag
gc.collect()
#统计在24小时内的平均month_lag
hour_installment = pd.pivot_table(ht,values='installments',index='card_id',columns=['purchase_hour'],aggfunc=np.sum,fill_value=0)
hour_installment.columns = ['month_installments_%s'%str(i) for i in range(24)]
hour_installment.reset_index(inplace=True)
train = pd.merge(train,hour_installment,on='card_id',how='left')
test = pd.merge(test,hour_installment,on='card_id',how='left')
del hour_installment
gc.collect()
#统计历史上总共买过多少次东西
ht_cnt = ht.groupby('card_id')['city_id'].count().reset_index()
ht_cnt.columns = ['card_id','ht_cnt']
#merge the result
train = pd.merge(train,ht_cnt,on='card_id',how='left')
test = pd.merge(test,ht_cnt,on='card_id',how='left')
del ht_cnt
gc.collect()
#计算历史上的消费状况
ht_amount = ht.groupby('card_id').agg({'purchase_amount':[np.max,np.mean,np.std]})
ht_amount.columns = ['amount_max','amount_mean','amount_std']
ht_amount = ht_amount.reset_index()
train = pd.merge(train,ht_amount,on='card_id',how='left')
test = pd.merge(test,ht_amount,on='card_id',how='left')
del ht_amount
gc.collect()
#计算历史上month_lag的情况
ht_month_lag = ht.groupby('card_id').agg({'month_lag':[np.min,np.mean,np.std]})
ht_month_lag.columns = ['lag_min','lag_mean','lag_std']
ht_month_lag = ht_month_lag.reset_index()
train = pd.merge(train,ht_month_lag,on='card_id',how='left')
test = pd.merge(test,ht_month_lag,on='card_id',how='left')
del ht_month_lag
gc.collect()
#计算历史上installments的情况
ht_installment = ht.groupby('card_id').agg({'installments':[np.max,np.min,np.mean,np.std]})
ht_installment.columns = ['ins_max','ins_min','ins_mean','ins_std']
ht_installment = ht_installment.reset_index()
train = pd.merge(train,ht_installment,on='card_id',how='left')
test = pd.merge(test,ht_installment,on='card_id',how='left')
del ht_installment
gc.collect()
ht_auth_flag = pd.pivot_table(ht,values='city_id',index='card_id',columns='authorized_flag',aggfunc=len,fill_value=0)
ht_auth_flag.columns = ['auth_y','auth_n']
ht_auth_flag = ht_auth_flag.reset_index()
train = pd.merge(train,ht_auth_flag,on='card_id',how='left')
test = pd.merge(test,ht_auth_flag,on='card_id',how='left')
del ht_auth_flag
gc.collect()
ht['merchant_category_id'] = ht['merchant_category_id'].map(lambda x:str(x))
ht_mer_cate = ht.groupby('card_id').apply(lambda x:' '.join(x['merchant_category_id']))
ht_mer_cate = ht_mer_cate.reset_index()
ht_mer_cate.columns = ['card_id','mer_cate_id_list']
train = pd.merge(train,ht_mer_cate,on='card_id',how='left')
test = pd.merge(test,ht_mer_cate,on='card_id',how='left')
del ht_mer_cate
gc.collect()
ht_state_cnt = ht.groupby('card_id')['state_id'].nunique().reset_index()
ht_state_cnt.columns = ['card_id','state_cnt']
train = pd.merge(train,ht_state_cnt,on='card_id',how='left')
test = pd.merge(test,ht_state_cnt,on='card_id',how='left')
del ht_state_cnt
gc.collect()
ht_mer_cate_cnt = ht.groupby('card_id')['merchant_category_id'].nunique().reset_index()
ht_mer_cate_cnt.columns = ['card_id','merchant_category_id_cnt']
train = pd.merge(train,ht_mer_cate_cnt,on='card_id',how='left')
test = pd.merge(test,ht_mer_cate_cnt,on='card_id',how='left')
del ht_mer_cate_cnt
gc.collect()
ht_mer_cnt = ht.groupby('card_id')['merchant_id'].nunique().reset_index()
ht_mer_cnt.columns = ['card_id','merchant_id_cnt']
train = pd.merge(train,ht_mer_cnt,on='card_id',how='left')
test = pd.merge(test,ht_mer_cnt,on='card_id',how='left')
del ht_mer_cnt
gc.collect()
#统计在新商店的24小时的购买记录次数分布
new_hour_purchase_cnt = pd.pivot_table(nmt,values='month_lag',index='card_id',columns=['purchase_hour'],aggfunc=len,fill_value=0)
new_hour_purchase_cnt.columns = ['new_purchase_hour_%s'%str(i) for i in range(24)]
new_hour_purchase_cnt.reset_index(inplace=True)
train = pd.merge(train,new_hour_purchase_cnt,on='card_id',how='left')
test = pd.merge(test,new_hour_purchase_cnt,on='card_id',how='left')
del new_hour_purchase_cnt
gc.collect()
#统计在24小时内的购买记录次数
new_hour_sub_id = pd.pivot_table(nmt,values='month_lag',index='card_id',columns=['subsector_id'],aggfunc=len,fill_value=0)
new_hour_sub_id.columns = ['new_sub_id_%s'%str(i) for i in range(nmt['subsector_id'].nunique())]
new_hour_sub_id.reset_index(inplace=True)
train = pd.merge(train,new_hour_sub_id,on='card_id',how='left')
test = pd.merge(test,new_hour_sub_id,on='card_id',how='left')

del new_hour_sub_id
gc.collect()
#统计在历史上每个月的购买次数
new_ym_purchase_cnt = pd.pivot_table(nmt,values='month_lag',index='card_id',columns=['purchase_year','purchase_month'],aggfunc=len,fill_value=0)
new_ym_purchase_cnt.columns = ['new_purchase_ym_%s'%str(i) for i in range(new_ym_purchase_cnt.shape[1])]
new_ym_purchase_cnt.reset_index(inplace=True)
train = pd.merge(train,new_ym_purchase_cnt,on='card_id',how='left')
test = pd.merge(test,new_ym_purchase_cnt,on='card_id',how='left')
del new_ym_purchase_cnt
gc.collect()
#统计在24小时内的平均购买金额
nmt_hour_purchase_amount = pd.pivot_table(nmt,values='purchase_amount',index='card_id',columns=['purchase_hour'],aggfunc=np.mean,fill_value=0)
nmt_hour_purchase_amount.columns = ['new_purchase_amount_%s'%str(i) for i in range(24)]
nmt_hour_purchase_amount.reset_index(inplace=True)
train = pd.merge(train,nmt_hour_purchase_amount,on='card_id',how='left')
test = pd.merge(test,nmt_hour_purchase_amount,on='card_id',how='left')
del nmt_hour_purchase_amount
gc.collect()
#统计在24小时内的平均购买金额
nmt_hour_month_lag = pd.pivot_table(nmt,values='month_lag',index='card_id',columns=['purchase_hour'],aggfunc=np.mean,fill_value=0)
nmt_hour_month_lag.columns = ['new_month_lag_%s'%str(i) for i in range(24)]
nmt_hour_month_lag.reset_index(inplace=True)
train = pd.merge(train,nmt_hour_month_lag,on='card_id',how='left')
test = pd.merge(test,nmt_hour_month_lag,on='card_id',how='left')
del nmt_hour_month_lag
gc.collect()
#统计在24小时内的平均购买金额
nmt_hour_installments = pd.pivot_table(nmt,values='installments',index='card_id',columns=['purchase_hour'],aggfunc=np.mean,fill_value=0)
nmt_hour_installments.columns = ['new_installments_%s'%str(i) for i in range(24)]
nmt_hour_installments.reset_index(inplace=True)
train = pd.merge(train,nmt_hour_installments,on='card_id',how='left')
test = pd.merge(test,nmt_hour_installments,on='card_id',how='left')
del nmt_hour_installments
gc.collect()
#统计历史上总共买过多少次东西
nmt_cnt = nmt.groupby('card_id')['city_id'].count().reset_index()
nmt_cnt.columns = ['card_id','nmt_cnt']
#merge the result
train = pd.merge(train,nmt_cnt,on='card_id',how='left')
test = pd.merge(test,nmt_cnt,on='card_id',how='left')
del nmt_cnt
gc.collect()
#计算历史上的消费状况
nmt_amount = nmt.groupby('card_id').agg({'purchase_amount':[np.max,np.mean,np.std]})
nmt_amount.columns = ['nmt_amount_max','nmt_amount_mean','nmt_amount_std']
nmt_amount = nmt_amount.reset_index()
train = pd.merge(train,nmt_amount,on='card_id',how='left')
test = pd.merge(test,nmt_amount,on='card_id',how='left')
del nmt_amount
gc.collect()
#计算历史上month_lag的情况
nmt_month_lag = nmt.groupby('card_id').agg({'month_lag':[np.min,np.mean,np.std]})
nmt_month_lag.columns = ['nmt_lag_min','nmt_lag_mean','nmt_lag_std']
nmt_month_lag = nmt_month_lag.reset_index()
train = pd.merge(train,nmt_month_lag,on='card_id',how='left')
test = pd.merge(test,nmt_month_lag,on='card_id',how='left')
del nmt_month_lag
gc.collect()
#计算历史上installments的情况
nmt_installment = nmt.groupby('card_id').agg({'installments':[np.max,np.min,np.mean,np.std]})
nmt_installment.columns = ['nmt_ins_max','nmt_ins_min','nmt_ins_mean','nmt_ins_std']
nmt_installment = nmt_installment.reset_index()
train = pd.merge(train,nmt_installment,on='card_id',how='left')
test = pd.merge(test,nmt_installment,on='card_id',how='left')
del nmt_installment
gc.collect()
nmt_auth_flag = pd.pivot_table(ht,values='city_id',index='card_id',columns='authorized_flag',aggfunc=len,fill_value=0)
nmt_auth_flag.columns = ['nmt_auth_y','nmt_auth_n']
nmt_auth_flag = nmt_auth_flag.reset_index()
train = pd.merge(train,nmt_auth_flag,on='card_id',how='left')
test = pd.merge(test,nmt_auth_flag,on='card_id',how='left')
del nmt_auth_flag
gc.collect()
nmt['merchant_category_id'] = ht['merchant_category_id'].map(lambda x:str(x))
nmt_mer_cate = nmt.groupby('card_id').apply(lambda x:' '.join(x['merchant_category_id']))
nmt_mer_cate = nmt_mer_cate.reset_index()
nmt_mer_cate.columns = ['card_id','nmt_mer_cate_id_list']
train = pd.merge(train,nmt_mer_cate,on='card_id',how='left')
test = pd.merge(test,nmt_mer_cate,on='card_id',how='left')
train['nmt_mer_cate_id_list'] = train['nmt_mer_cate_id_list'].fillna('')
test['nmt_mer_cate_id_list'] = test['nmt_mer_cate_id_list'].fillna('')
del nmt_mer_cate
gc.collect()
nmt_state_cnt = nmt.groupby('card_id')['state_id'].nunique().reset_index()
nmt_state_cnt.columns = ['card_id','nmt_state_cnt']
train = pd.merge(train,nmt_state_cnt,on='card_id',how='left')
test = pd.merge(test,nmt_state_cnt,on='card_id',how='left')
del nmt_state_cnt
gc.collect()
nmt_mer_cate_cnt = nmt.groupby('card_id')['merchant_category_id'].nunique().reset_index()
nmt_mer_cate_cnt.columns = ['card_id','nmt_merchant_category_id_cnt']
train = pd.merge(train,nmt_mer_cate_cnt,on='card_id',how='left')
test = pd.merge(test,nmt_mer_cate_cnt,on='card_id',how='left')
del nmt_mer_cate_cnt
gc.collect()
nmt_mer_cnt = nmt.groupby('card_id')['merchant_id'].nunique().reset_index()
nmt_mer_cnt.columns = ['card_id','nmt_merchant_id_cnt']
train = pd.merge(train,nmt_mer_cnt,on='card_id',how='left')
test = pd.merge(test,nmt_mer_cnt,on='card_id',how='left')
del nmt_mer_cnt
gc.collect()
train_cate = pd.get_dummies(train[['feature_1','feature_2','feature_3']])
test_cate = pd.get_dummies(test[['feature_1','feature_2','feature_3']])
cv = CountVectorizer()
cv_train = cv.fit_transform(train['mer_cate_id_list'])
cv_test =  cv.transform(test['mer_cate_id_list'])
train.drop('mer_cate_id_list',axis=1,inplace=True)
test.drop('mer_cate_id_list',axis=1,inplace=True)
ss = StandardScaler(with_mean=False)
cv_train = ss.fit_transform(cv_train)
cv_test = ss.transform(cv_test)
cv_ = CountVectorizer()
cv_train_ = cv_.fit_transform(train['nmt_mer_cate_id_list'])
cv_test_ =  cv_.transform(test['nmt_mer_cate_id_list'])
train.drop('nmt_mer_cate_id_list',axis=1,inplace=True)
test.drop('nmt_mer_cate_id_list',axis=1,inplace=True)
ss = StandardScaler(with_mean=False)
cv_train_ = ss.fit_transform(cv_train_)
cv_test_ = ss.transform(cv_test_)
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
#handler the numeric feature
drop_columns = ['first_active_month','card_id','feature_1','feature_2','feature_3','target','year','month','cate']
num_feature = [v for v in train.columns if v not in drop_columns]
train_num_feature = train[num_feature].values
test_num_feature = test[num_feature].values
ss = StandardScaler()
train_num_feature = ss.fit_transform(train_num_feature)
test_num_feature = ss.transform(test_num_feature)
#stack the feature
train_feature = hstack([train_num_feature,cv_train,cv_train_,train_cate]).tocsr()
test_feature = hstack([test_num_feature,cv_test,cv_test_,test_cate]).tocsr()
del train_num_feature
del test_num_feature
gc.collect()
#不用stacking
res_lgb = []
feature_imp = []
for tr,va in KFold(n_splits=10,random_state=2018).split(train_feature,train['target'].values):
    lgbmr = LGBMRegressor(num_leaves=32,n_estimators=150,colsample_bytree=0.7,subsample=0.7)
    lgbmr.fit(train_feature[tr],train['target'].values[tr],early_stopping_rounds=10,
              eval_set=[(train_feature[tr],train['target'].values[tr]),(train_feature[va],train['target'].values[va])],
              eval_metric='rmse',
              verbose=50)
    feature_imp.append(lgbmr.feature_importances_)
    res_lgb.append(lgbmr.predict(test_feature))
f = np.mean(feature_imp,axis=0)
avg_res = np.mean(res_lgb,axis=0)
test['target'] = avg_res
test[['card_id','target']].to_csv('predictlgb_1225.csv',index=False)
