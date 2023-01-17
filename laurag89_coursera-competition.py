import numpy as np
import pandas as pd

parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')

train = pd.read_csv('../input/sales_train.csv.gz',compression = 'gzip', parse_dates=['date'], date_parser=parser)
test = pd.read_csv('../input/test.csv.gz',compression = 'gzip')
submission = pd.read_csv('../input/sample_submission.csv.gz', compression = 'gzip')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
test_shops = test.shop_id.unique()
train = train[train.shop_id.isin(test_shops)]
test_items = test.item_id.unique()
train = train[train.item_id.isin(test_items)]

print('train:', train.shape, 'test:', test.shape, 'items:', items.shape, 'shops:', shops.shape)
test_only = test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month

train.head()
train['date_block_num'].max()
# group by 
train_grp = train.groupby(['date_block_num','shop_id','item_id'])
# price mean by month
train_price = pd.DataFrame(train_grp.mean()['item_price']).reset_index()
train_price.head()
# count summary by month
train_monthly = pd.DataFrame(train_grp.sum()['item_cnt_day']).reset_index()
train_monthly.rename(columns={'item_cnt_day':'item_cnt'}, inplace=True)
train_monthly.head()
train_piv = train_monthly.pivot_table(index=['shop_id','item_id'], columns=['date_block_num'], values='item_cnt', aggfunc=np.sum, fill_value=0)
train_piv = train_piv.reset_index()
train_piv.head()
# By shop
grp = train_monthly.groupby(['shop_id', 'item_id'])
train_shop = grp.agg({'item_cnt':['mean','median','std']}).reset_index()
train_shop.columns = ['shop_id','item_id','cnt_mean_shop','cnt_med_shop','cnt_std_shop']
train_shop.head()
# By shop&category
train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')
grp = train_cat_monthly.groupby(['shop_id', 'item_category_id'])
train_shop_cat = grp.agg({'item_cnt':['mean']}).reset_index()
train_shop_cat.columns = ['shop_id','item_category_id','cnt_mean_cat_shop']
train_shop_cat.head()
# Last month
train_last = train_monthly[train_monthly['date_block_num'] == 33]
train_last = train_last.drop(['date_block_num'], axis=1).rename(columns={'item_cnt':'cnt_sum_last'})
train_last.head()
# Prev month
train_prev = train_monthly.copy()
train_prev['date_block_num'] = train_prev['date_block_num'] + 1
train_prev = train_prev.rename(columns={'item_cnt':'cnt_sum_prev'})
train_prev.head()
# Prev month by category
train_cat_prev = pd.merge(train_prev, items, on=['item_id'], how='left')
grp = train_cat_prev.groupby(['date_block_num','shop_id','item_category_id'])
train_cat_prev = grp['cnt_sum_prev'].sum().reset_index()
train_cat_prev = train_cat_prev.rename(columns={'cnt_sum_prev':'cnt_sum_cat_prev'})
train_cat_prev.head()
# Prev month EMA,MACD,SIG
col = np.arange(34)
pivT = train_piv[col].T
evm_s = pivT.ewm(span=12).mean().T
evm_l = pivT.ewm(span=26).mean().T
macd = evm_s - evm_l
sig = macd.ewm(span=9).mean()

train_piv_key = train_piv.loc[:,['shop_id','item_id']]
train_evm_list = []
for c in col:
  sub_evm_s = pd.DataFrame(evm_s.loc[:,c]).rename(columns={c:'cnt_evm_s_prev'})
  sub_evm_l = pd.DataFrame(evm_l.loc[:,c]).rename(columns={c:'cnt_evm_l_prev'})
  sub_macd = pd.DataFrame(evm_l.loc[:,c]).rename(columns={c:'cnt_macd_prev'})
  sub_sig = pd.DataFrame(evm_l.loc[:,c]).rename(columns={c:'cnt_sig_prev'})
  
  sub_evm = pd.concat([train_piv_key, sub_evm_s, sub_evm_l, sub_macd, sub_sig], axis=1)
  sub_evm['date_block_num'] = c + 1
  train_evm_list.append(sub_evm)
    
train_evm_prev = pd.concat(train_evm_list)
#train_evm_prev.head()
train_evm_prev.query("shop_id == 2 & item_id == 30").tail()
item_cats['item_category_group'] = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])
item_cats['item_category_group'] = pd.Categorical(item_cats['item_category_group']).codes
item_cats = pd.merge(item_cats, pd.get_dummies(item_cats['item_category_group'], prefix='item_category_group', drop_first=True), left_index=True, right_index=True)
item_cats.drop(['item_category_group'], axis=1, inplace=True)

shops['city'] = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
shops['city'] = pd.Categorical(shops['city']).codes
def mergeFeature(source): 
  d = source
  d = pd.merge(d, items, on=['item_id'], how='left')
  d = pd.merge(d, item_cats, on=['item_category_id'], how='left')
  d = pd.merge(d, shops, on=['shop_id'], how='left')

  d = pd.merge(d, train_price, on=['date_block_num','shop_id','item_id'], how='left')
  d = pd.merge(d, train_shop, on=['shop_id','item_id'], how='left')
  #d = pd.merge(d, train_shop_cat, on=['shop_id','item_category_id'], how='left')
  #d = pd.merge(d, train_last, on=['shop_id','item_id'], how='left')
  d = pd.merge(d, train_prev, on=['date_block_num','shop_id','item_id'], how='left')
  d = pd.merge(d, train_evm_prev, on=['date_block_num','shop_id','item_id'], how='left')
  d = pd.merge(d, train_cat_prev, on=['date_block_num','shop_id','item_category_id'], how='left')
  
  d.drop(['shop_id','shop_name','item_id','item_name','item_category_id','item_category_name'], axis=1, inplace=True)
  d.fillna(0.0, inplace=True)
  return d
train_set = mergeFeature(train_monthly)

X_train = train_set.drop(['item_cnt'], axis=1)
Y_train = train_set['item_cnt']

X_train.head()
test['date_block_num'] = 34

X_test = mergeFeature(test.drop(['ID'], axis=1))
X_test.head()
_ = '''
'''
import xgboost as xgb

reg = xgb.XGBRegressor(n_estimators=25, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=1, eval_metric='rmse')

reg.fit(X_train, Y_train)
pred_cnt = reg.predict(X_test)
result = pd.DataFrame({
    "ID": test["ID"],
    "item_cnt_month": pred_cnt.clip(0. ,20.)
})
result.to_csv("submission.csv", index=False)