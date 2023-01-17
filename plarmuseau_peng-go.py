import os
print(os.listdir("../input"))

import gc
import time
start_time = time.time()
import pandas as pd
import numpy as np


print('reading===========================================================================\n')
train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
end_time = time.time()
print(str(round(end_time-start_time, 2)) + 's' + '完成读写数据...')
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
print('train:', train.shape, 'test:', test.shape)
print('missingcol===========================================================================\n')
print('只在训练数据集里存在的特征：train coloms not in test')
print([col for col in train.columns if col not in test.columns])

print('tfidf')
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
ica = FastICA(n_components=10)


itemcatids=item_categories['item_category_id']
itemids=items['item_id']
itemcat=items['item_category_id']
shopsid=shops['shop_id']
feature_cnt = 3000
aant=10

tfidf = TfidfVectorizer( max_df=0.6,max_features=feature_cnt, ngram_range=(1, 2))


item_categories['item_category_name_len'] = item_categories['item_category_name'].apply(len)
item_categories['item_category_name_wc'] = item_categories['item_category_name'].apply(lambda x: len(str(x).split(' '))) 
txtFeatures = pd.DataFrame(tfidf.fit_transform(item_categories['item_category_name']).toarray())
#u, s, vh = np.linalg.svd(txtFeatures, full_matrices=False)
#item_categories=pd.DataFrame( u[:,:aant] * s[:aant] )  #pd.DataFrame(  np.dot(u[:,:aant] * s[:aant], vh[:aant,:]) )
S_ = ica.fit_transform(txtFeatures)  # Reconstruct signals
item_categories=pd.DataFrame(S_ )
#A_ = ica.mixing_  # Get estimated mixing matrix
item_categories['item_category_id']=itemcatids
print(item_categories.head())



items['item_name_len'] = items['item_name'].apply(len) #Lenth of Item Description
items['item_name_wc'] = items['item_name'].apply(lambda x: len(str(x).split(' '))) #Item Description Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
#u, s, vh = np.linalg.svd(txtFeatures, full_matrices=False)
#items=pd.DataFrame( u[:,:aant+30] * s[:aant+30] ) #pd.DataFrame(  np.dot(u[:,:aant+aant] * s[:aant+aant], vh[:aant+aant,:]) )
S_ = ica.fit_transform(txtFeatures)  # Reconstruct signals
items=pd.DataFrame(S_ )
items['item_id']=itemids
items['item_category_id']=itemcat
print(items.head())

shops['shop_name_len'] = shops['shop_name'].apply(len)  #Lenth of Shop Name
shops['shop_name_wc'] = shops['shop_name'].apply(lambda x: len(str(x).split(' '))) #Shop Name Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(shops['shop_name']).toarray())
#u, s, vh = np.linalg.svd(txtFeatures, full_matrices=False)
#shops=pd.DataFrame( u[:,:aant] * s[:aant] ) #pd.DataFrame(  np.dot(u[:,:aant] * s[:aant], vh[:aant,:]) )
S_ = ica.fit_transform(txtFeatures)  # Reconstruct signals
shops=pd.DataFrame(S_ )
shops['shop_id']=shopsid
print(shops.head())


end_time = time.time()
print(str(round(end_time-start_time, 2)) + 's' + '完成特征数据提取...')

print('将销售日期转换位标准格式...')
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
print(train.date.head())

train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
print('magic===========================================================================\n')

## MAGIC
## 每个月每个店每一件商品的销售量
print('统计每一月每一个店每一件商品的销售量...')
train = train.drop(['date','item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})
train.head()

#remove trend and added moving average
#changed difference to 1month
# moving average to 12month MA= deaseasonalised
#tr12=train[['shop_id','item_id','date_block_num','item_cnt_month']].sort_values(['shop_id','item_id','date_block_num']).diff(periods=12).drop(['shop_id','item_id','date_block_num'],axis=1).fillna(0)
#train['diff12']=tr12.sort_index()['item_cnt_month']
#tr1=train[['shop_id','item_id','date_block_num','item_cnt_month']].sort_values(['shop_id','item_id','date_block_num']).diff(periods=1).drop(['shop_id','item_id','date_block_num'],axis=1).fillna(0)
#train['diff1']=tr1.sort_index()['item_cnt_month']
#trma=train[['shop_id','item_id','date_block_num','item_cnt_month']].sort_values(['shop_id','item_id','date_block_num']).rolling(window=6).mean().drop(['shop_id','item_id','date_block_num'],axis=1).fillna(0)
#train['ma2']=trma.sort_index()['item_cnt_month']



print('===========================================================================\n')

# 不同月相同店相同商品销量
# print([train[train.shop_id == i][train.item_id == j] for i in range(shops.shape[0]) for j in train.item_id.unique()])
# changed to MEDIAN
print('统计每一个店每一件商品平均月销量...')
shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].median()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
print(shop_item_monthly_mean.head())

print('===========================================================================\n')
print('将每一个店每一件商品平均月销量这列特征与train合并...')
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])
print(train.head())



print('===========================================================================\n')
## 以左连接合并所有数据
#Last Month (Oct 2015)
print('以train数据集为中心合并所有数据...')
# shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
# shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})
# shop_item_prev_month.head()
# Add Previous Month Feature
# train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

print(train.head())
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_categories, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')
print(train.head())

print('===========================================================================\n')
### feature construction
## forecasting the sales for these shops and products for November 2015
# should made some changes and add some features
# just the same as train

print('以test数据集为中心合并所有数据...')
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34

#test = pd.merge(test, train[['shop_id','item_id','date_block_num','diff1','diff12','ma2']], how='left', on=['shop_id','item_id','date_block_num']).fillna(0.)
test = pd.merge(test, train[['shop_id','item_id','date_block_num']], how='left', on=['shop_id','item_id','date_block_num']).fillna(0.)
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)
test = pd.merge(test, items, how='left', on='item_id')
#Item Category features
test = pd.merge(test, item_categories, how='left', on='item_category_id')
#Shops features
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0.
print(test.head())
print(test.columns)
end_time = time.time()
print(str(round(end_time-start_time, 2)) + 's' + '完成特征工程...')


print('===========================================================================\n')
print('LGB模型构建...')
from sklearn.preprocessing import LabelEncoder

for col in train.columns:
    if train[col].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values) + list(test[col].values))
        train[col] = lbl.fit_transform(train[col].values)
        test[col] = lbl.fit_transform(test[col].values)

train['item_cnt_month'] = np.log1p(train['item_cnt_month'].clip(0.,20.))
cols = [c for c in train.columns if c not in ['item_cnt_month']]
x = train[cols]
y = train['item_cnt_month']

del item_categories
del items
del train
del shop_item_monthly_mean
del shops
del txtFeatures
gc.collect()

############
############
##LIGHTGBM##
############
############

from sklearn.model_selection import train_test_split
import lightgbm as lgb


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=64)

params = {
        'boosting_type': 'gbdt',    #采取梯度提升
        'objective': 'binary',    #application 选择回归方式
        'metric': 'l2_root',    # root_mean_squared_error
        'max_depth': 20,    
        'num_leaves': 30,    # 一棵树上的叶子数
        'learning_rate': 0.125,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,    # 在不进行重采样的情况下随机选择部分数据
        'bagging_freq': 5,
        'verbose': 1,    #=0 = 错误 (警告)
        'num_threads': 4,  # 线程数，与cpu核数有关，一核两线程
        'lambda_l2': 1,    # L1正则项
        'min_gain_to_split': 0,    # 执行切分的最小增益
        'seed':1234,
        'min_data': 28,    # 一个叶子上数据的最小数量，避免过拟合
        'min_hessian': 0.025    # 一个叶子上的最小 hessian 和,避免过拟合
        }

model = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(X_test, y_test)],
            early_stopping_rounds=100,
            verbose_eval=25)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

end_time = time.time()
print(str(round(end_time-start_time, 2)) + 's' + '完成LGB模型训练...')

test['item_cnt_month'] = model.predict(test[cols], num_iteration=model.best_iteration)
#test['item_cnt_month'] = np.exp(test.item_cnt_month)
test[['ID','item_cnt_month']].to_csv('submission.csv', index=False)

end_time = time.time()
print(str(round(end_time-start_time, 2)) + 's' + '完成LGB模型测试...')
print('===========================================================================')